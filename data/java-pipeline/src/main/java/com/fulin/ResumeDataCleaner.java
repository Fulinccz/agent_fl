package com.fulin;

import java.util.regex.Pattern;

public class ResumeDataCleaner {

    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final Pattern CONTROL_CHAR_PATTERN = Pattern.compile("[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]");
    private static final Pattern HTML_TAG_PATTERN = Pattern.compile("<[^>]+>");
    private static final Pattern HTML_ENTITY_PATTERN = Pattern.compile("&[a-zA-Z0-9#]+;");
    private static final Pattern MEANINGLESS_PATTERN = Pattern.compile("^[\\d\\s\\p{Punct}]+$");

    private static final int MAX_FIELD_LENGTH = 5000;
    private static final int MIN_MEANINGFUL_LENGTH = 10;

    public static String cleanText(String text) {
        if (text == null || text.isEmpty()) {
            return "";
        }

        // Remove HTML tags
        text = HTML_TAG_PATTERN.matcher(text).replaceAll("");

        // Remove HTML entities (&nbsp; &amp; etc.)
        text = HTML_ENTITY_PATTERN.matcher(text).replaceAll("");

        // Normalize whitespace
        text = WHITESPACE_PATTERN.matcher(text).replaceAll(" ");

        // Remove control characters
        text = CONTROL_CHAR_PATTERN.matcher(text).replaceAll("");

        // Trim
        text = text.trim();

        // Limit length
        if (text.length() > MAX_FIELD_LENGTH) {
            text = text.substring(0, MAX_FIELD_LENGTH);
        }

        return text;
    }

    public static boolean validate(Resume resume) {
        if (resume == null) {
            return false;
        }

        String work = resume.getWorkDescription();
        String project = resume.getProjectDescription();

        // Must have at least one non-empty field
        if ((work == null || work.isEmpty()) && (project == null || project.isEmpty())) {
            return false;
        }

        // Check if content is meaningful (not just numbers/symbols)
        String combined = (work == null ? "" : work) + (project == null ? "" : project);
        if (combined.length() < MIN_MEANINGFUL_LENGTH) {
            return false;
        }

        if (MEANINGLESS_PATTERN.matcher(combined).matches()) {
            return false;
        }

        // Check Chinese character ratio (at least 20% should be Chinese)
        long chineseCount = combined.chars().filter(c -> c >= 0x4E00 && c <= 0x9FFF).count();
        if (chineseCount < 2 && combined.length() > 20) {
            return false;
        }

        return true;
    }
}
