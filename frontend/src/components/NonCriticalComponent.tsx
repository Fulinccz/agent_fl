import React from 'react';

const NonCriticalComponent: React.FC = () => {
  return (
    <div className="non-critical-component">
      <div className="divider"></div>
      <h2>How to Use</h2>
      <div className="project-info">
        <div className="project-item">
          <h3>About This Project</h3>
          <p>This is an AI Resume Intelligent Optimization Platform that helps you improve your resume using advanced AI technologies. The platform integrates resume parsing, JD matching, intelligent optimization, and content rewriting capabilities.</p>
        </div>
        <div className="project-item">
          <h3>Key Features</h3>
          <ul>
            <li>AI-powered resume optimization</li>
            <li>JD matching and analysis</li>
            <li>Intelligent content rewriting</li>
            <li>File upload support for resume analysis</li>
            <li>Conversation history management</li>
          </ul>
        </div>
        <div className="project-item">
          <h3>How to Use</h3>
          <ol>
            <li>Enter your query in the text input box</li>
            <li>Optional: Upload a file by clicking the attachment button</li>
            <li>Click the send button to get AI-generated responses</li>
            <li>View the AI's thinking process in the left panel</li>
            <li>See the final result in the right panel</li>
            <li>Check your conversation history by clicking "View History"</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default NonCriticalComponent;