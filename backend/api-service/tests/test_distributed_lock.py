import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from cache.distributed_lock import DistributedLock, distributed_lock, LockAcquireError


class TestDistributedLock:
    """分布式锁测试套件"""

    @pytest.fixture
    def mock_redis(self):
        mock_client = AsyncMock()
        return mock_client

    @pytest.fixture
    def lock(self, mock_redis):
        return DistributedLock(redis_client=mock_redis)

    # ---- 基础获取/释放测试 ----

    @pytest.mark.asyncio
    async def test_acquire_returns_token(self, lock, mock_redis):
        """成功获取锁应返回 UUID token"""
        mock_redis.set.return_value = True
        token = await lock.acquire("test:lock")
        assert token is not None
        assert len(token) > 0
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_sets_nx_and_ex(self, lock, mock_redis):
        """获取锁应使用 NX + EX 参数"""
        mock_redis.set.return_value = True
        await lock.acquire("test:lock", timeout=30)
        call_kwargs = mock_redis.set.call_args[1]
        assert call_kwargs["nx"] is True
        assert call_kwargs["ex"] == 30

    @pytest.mark.asyncio
    async def test_acquire_with_custom_timeout(self, lock, mock_redis):
        """自定义超时时间应正确传递"""
        mock_redis.set.return_value = True
        await lock.acquire("test:lock", timeout=120)
        call_kwargs = mock_redis.set.call_args[1]
        assert call_kwargs["ex"] == 120

    @pytest.mark.asyncio
    async def test_acquire_block_false_when_locked(self, lock, mock_redis):
        """非阻塞模式下锁被占用应立即返回 None"""
        mock_redis.set.return_value = None
        result = await lock.acquire("test:lock", block=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_release_success(self, lock, mock_redis):
        """持有者释放锁应返回 True"""
        mock_redis.set.return_value = "some-token"
        mock_redis.eval.return_value = 1
        token = await lock.acquire("test:lock")
        result = await lock.release("test:lock", token)
        assert result is True

    @pytest.mark.asyncio
    async def test_release_non_owner_fails(self, lock, mock_redis):
        """非持有者释放锁应返回 False"""
        mock_redis.eval.return_value = 0
        result = await lock.release("test:lock", "wrong-token")
        assert result is False

    # ---- Lua 脚本原子性测试 ----

    @pytest.mark.asyncio
    async def test_release_uses_lua_script(self, lock, mock_redis):
        """释放锁应使用 Lua 脚本保证原子性"""
        mock_redis.set.return_value = "token"
        mock_redis.eval.return_value = 1
        token = await lock.acquire("test:lock")
        await lock.release("test:lock", token)
        lua_script = mock_redis.eval.call_args[0][0]
        assert "get" in lua_script
        assert "del" in lua_script
        assert "ARGV[1]" in lua_script

    # ---- 延长锁时间测试 ----

    @pytest.mark.asyncio
    async def test_extend_success(self, lock, mock_redis):
        """延长锁时间应成功"""
        mock_redis.set.return_value = "token"
        mock_redis.eval.return_value = 1
        token = await lock.acquire("test:lock")
        result = await lock.extend("test:lock", token, 60)
        assert result is True

    @pytest.mark.asyncio
    async def test_extend_non_owner_fails(self, lock, mock_redis):
        """非持有者延长锁应失败"""
        mock_redis.eval.return_value = 0
        result = await lock.extend("test:lock", "wrong-token", 60)
        assert result is False

    # ---- 错误处理测试 ----

    @pytest.mark.asyncio
    async def test_acquire_redis_error(self, lock, mock_redis):
        """Redis 异常时获取锁应返回 None"""
        from redis.exceptions import RedisError
        mock_redis.set.side_effect = RedisError("Connection refused")
        result = await lock.acquire("test:lock", block=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_release_redis_error(self, lock, mock_redis):
        """Redis 异常时释放锁应返回 False"""
        from redis.exceptions import RedisError
        mock_redis.eval.side_effect = RedisError("Connection failed")
        result = await lock.release("test:lock", "token")
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_redis_error(self, lock, mock_redis):
        """Redis 异常时延长锁应返回 False"""
        from redis.exceptions import RedisError
        mock_redis.eval.side_effect = RedisError("Error")
        result = await lock.extend("test:lock", "token", 30)
        assert result is False

    # ---- 阻塞等待测试 ----

    @pytest.mark.asyncio
    async def test_blocking_acquire_eventually_succeeds(self, lock, mock_redis):
        """阻塞模式下最终获取到锁应返回 token"""
        call_count = [0]

        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                return None
            return True

        mock_redis.set.side_effect = side_effect
        token = await lock.acquire("test:lock", block=True)
        assert token is not None
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_blocking_acquire_timeout(self, lock, mock_redis):
        """阻塞模式超时应返回 None"""
        mock_redis.set.return_value = None
        result = await lock.acquire("test:lock", block=True)
        assert result is None


class TestDistributedLockContextManager:
    """分布式锁上下文管理器测试套件"""

    @pytest.mark.asyncio
    async def test_context_manager_acquires_and_releases(self):
        """上下文管理器应在进入和退出时自动获取/释放锁"""
        with patch("cache.distributed_lock._lock") as mock_lock_instance:
            mock_lock_instance.acquire = AsyncMock(return_value="test-token")
            mock_lock_instance.release = AsyncMock(return_value=True)

            async with distributed_lock("test:key", timeout=30) as token:
                assert token == "test-token"

            mock_lock_instance.acquire.assert_called_once()
            mock_lock_instance.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_raises_on_failure(self):
        """获取锁失败应抛出 LockAcquireError"""
        with patch("cache.distributed_lock._lock") as mock_lock_instance:
            mock_lock_instance.acquire = AsyncMock(return_value=None)
            with pytest.raises(LockAcquireError):
                async with distributed_lock("test:key"):
                    pass

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        """异常发生时上下文管理器仍应释放锁"""
        with patch("cache.distributed_lock._lock") as mock_lock_instance:
            mock_lock_instance.acquire = AsyncMock(return_value="test-token")
            mock_lock_instance.release = AsyncMock(return_value=True)

            with pytest.raises(ValueError):
                async with distributed_lock("test:key"):
                    raise ValueError("业务逻辑错误")

            mock_lock_instance.release.assert_called_once()


class TestDistributedLockDefaultInstance:
    """全局默认实例测试"""

    def test_default_instance_exists(self):
        """模块级 _lock 实例应存在"""
        from cache.distributed_lock import _lock
        assert isinstance(_lock, DistributedLock)

    def test_default_instance_lazy_init(self):
        """默认实例应延迟初始化 Redis 连接"""
        from cache.distributed_lock import _lock
        assert _lock.redis is None
