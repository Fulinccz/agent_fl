package com.fulin;

import java.io.Serializable;

public class Resume implements Serializable {
    private String resumeId;
    private String degree;
    private String universityType;
    private String workDescription;
    private String projectDescription;

    public Resume() {}

    public String getResumeId() { return resumeId; }
    public void setResumeId(String resumeId) { this.resumeId = resumeId; }

    public String getDegree() { return degree; }
    public void setDegree(String degree) { this.degree = degree; }

    public String getUniversityType() { return universityType; }
    public void setUniversityType(String universityType) { this.universityType = universityType; }

    public String getWorkDescription() { return workDescription; }
    public void setWorkDescription(String workDescription) { this.workDescription = workDescription; }

    public String getProjectDescription() { return projectDescription; }
    public void setProjectDescription(String projectDescription) { this.projectDescription = projectDescription; }

    public String getHash() {
        String content = (workDescription == null ? "" : workDescription) +
                        (projectDescription == null ? "" : projectDescription);
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            return String.valueOf(content.hashCode());
        }
    }

    public String toText() {
        StringBuilder sb = new StringBuilder();
        if (degree != null && !degree.isEmpty()) {
            sb.append("学历: ").append(degree).append("\n");
        }
        if (universityType != null && !universityType.isEmpty()) {
            sb.append("院校: ").append(universityType).append("\n");
        }
        if (workDescription != null && !workDescription.isEmpty()) {
            sb.append("工作经历: ").append(workDescription).append("\n");
        }
        if (projectDescription != null && !projectDescription.isEmpty()) {
            sb.append("项目经历: ").append(projectDescription).append("\n");
        }
        return sb.toString().trim();
    }

    @Override
    public String toString() {
        return "Resume{resumeId='" + resumeId + "', degree='" + degree + "', universityType='" + universityType + "'}";
    }
}
