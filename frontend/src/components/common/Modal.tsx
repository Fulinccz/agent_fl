import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消'
}) => {
  if (!isOpen) return null;

  return (
    <div 
      className="modal-overlay"
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <p>{message}</p>
        <div className="modal-actions">
          <button 
            className="modal-cancel"
            onClick={onClose}
          >
            {cancelText}
          </button>
          <button 
            className="modal-confirm"
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Modal;
