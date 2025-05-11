# AI Interviewer API Documentation

## Overview

The AI Interviewer API provides endpoints for creating and managing automated interview sessions. The system uses LLaMA 3.3-70B to conduct technical and behavioral interviews for job candidates, providing a realistic interview experience.

**Base URL**: `http://your-server-address:8000`

## Authentication

Currently, the API does not require authentication. Authentication mechanisms should be implemented for production deployments.

## Models

### InterviewSettings

```typescript
interface InterviewSettings {
  temperature: number;        // Default: 0.7
  top_p: number;              // Default: 0.9
  skills: string[];           // Default: []
  job_position: string;       // Default: ""
  job_description: string;    // Default: ""
  technical_questions: number; // Default: 5
  behavioral_questions: number; // Default: 3
  custom_questions?: string[]; // Optional
  candidate_name: string;     // Default: ""
}
```

### ChatMessage

```typescript
interface ChatMessage {
  message: string;
  session_id: string;
  type: "user" | "system";  // Default: "user"
}
```

### SessionResponse

```typescript
interface SessionResponse {
  session_id: string;
  message: string;
}
```

### InterviewStatus

```typescript
interface InterviewStatus {
  session_id: string;
  active: boolean;
  questions_asked: number;
  total_expected_questions: number;
}
```

## Endpoints

### Create a Session

Creates a new interview session with the specified settings.

**Endpoint**: `POST /create-session`

**Request Body**: `InterviewSettings` object (optional)

**Response**: 
```json
{
  "session_id": "uuid-string",
  "message": "Session created successfully"
}
```

**Notes**:
- The model name is fixed to "llama-3.3-70b-versatile" and cannot be changed
- All settings are optional and will use defaults if not provided
- The session is initialized but the interview is not automatically started

**Example**:
```javascript
// Nest.js implementation
@Post('create-session')
async createSession(@Body() settings: InterviewSettings): Promise<SessionResponse> {
  const response = await this.httpService.post(`${API_URL}/create-session`, settings).toPromise();
  return response.data;
}
```

### Start an Interview

Starts an interview in the specified session.

**Endpoint**: `POST /start-interview/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: Streamed text response

**Notes**:
- Returns a streaming response with the AI's introduction
- If an interview is already in progress, returns a 400 error

**Example**:
```javascript
// Nest.js implementation
@Post('start-interview/:sessionId')
async startInterview(@Param('sessionId') sessionId: string, @Res() res: Response) {
  const apiResponse = await this.httpService.post(
    `${API_URL}/start-interview/${sessionId}`,
    {},
    { responseType: 'stream' }
  ).toPromise();
  
  apiResponse.data.pipe(res);
}
```

### Send a Message

Sends a message from the candidate or a system command in an existing session.

**Endpoint**: `POST /chat/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Request Body**: `ChatMessage` object

**Response**: Streamed text response from AI

**Notes**:
- Returns a streaming response with the AI's reply
- Message type can be "user" (candidate) or "system" (admin commands)
- If no active interview in the session, returns a 400 error

**Example**:
```javascript
// Nest.js implementation
@Post('chat/:sessionId')
async chat(@Param('sessionId') sessionId: string, @Body() chatMessage: ChatMessage, @Res() res: Response) {
  // Ensure session_id in URL matches the one in body
  if (sessionId !== chatMessage.session_id) {
    throw new BadRequestException('Session ID mismatch');
  }
  
  const apiResponse = await this.httpService.post(
    `${API_URL}/chat/${sessionId}`,
    chatMessage,
    { responseType: 'stream' }
  ).toPromise();
  
  apiResponse.data.pipe(res);
}
```

### Reset a Conversation

Resets the conversation for a specific session, allowing a new interview to be started.

**Endpoint**: `POST /reset/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: 
```json
{
  "session_id": "uuid-string",
  "message": "Conversation reset. Ready to start a new interview."
}
```

**Example**:
```javascript
// Nest.js implementation
@Post('reset/:sessionId')
async resetConversation(@Param('sessionId') sessionId: string): Promise<SessionResponse> {
  const response = await this.httpService.post(`${API_URL}/reset/${sessionId}`).toPromise();
  return response.data;
}
```

### Get Interview Status

Checks if an interview is in progress and gets question counts.

**Endpoint**: `GET /interview-status/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: `InterviewStatus` object
```json
{
  "session_id": "uuid-string",
  "active": true,
  "questions_asked": 2,
  "total_expected_questions": 8
}
```

**Example**:
```javascript
// Nest.js implementation
@Get('interview-status/:sessionId')
async getInterviewStatus(@Param('sessionId') sessionId: string): Promise<InterviewStatus> {
  const response = await this.httpService.get(`${API_URL}/interview-status/${sessionId}`).toPromise();
  return response.data;
}
```

### Get Conversation History

Gets the complete conversation history for a session.

**Endpoint**: `GET /conversation-history/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**:
```json
{
  "session_id": "uuid-string",
  "model_name": "llama-3.3-70b-versatile",
  "active": true,
  "questions_asked": 3,
  "total_expected_questions": 8,
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, I am Mia, your interviewer..."}
  ]
}
```

**Example**:
```javascript
// Nest.js implementation
@Get('conversation-history/:sessionId')
async getConversationHistory(@Param('sessionId') sessionId: string) {
  const response = await this.httpService.get(`${API_URL}/conversation-history/${sessionId}`).toPromise();
  return response.data;
}
```

### Save Session

Saves the session conversation to a file on the server.

**Endpoint**: `POST /save-session/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: 
```json
{
  "session_id": "uuid-string",
  "message": "Session saving initiated"
}
```

**Notes**:
- The saving process happens in the background
- The file is saved on the server in the "interview_sessions" directory

**Example**:
```javascript
// Nest.js implementation
@Post('save-session/:sessionId')
async saveSession(@Param('sessionId') sessionId: string): Promise<SessionResponse> {
  const response = await this.httpService.post(`${API_URL}/save-session/${sessionId}`).toPromise();
  return response.data;
}
```

### List Sessions

Lists all active sessions and their status.

**Endpoint**: `GET /sessions`

**Response**:
```json
{
  "uuid-string-1": {
    "model_name": "llama-3.3-70b-versatile",
    "last_access": "2024-05-12T15:30:45.123456",
    "interview_active": true,
    "questions_asked": 3,
    "total_expected_questions": 8
  },
  "uuid-string-2": {
    "model_name": "llama-3.3-70b-versatile",
    "last_access": "2024-05-12T14:20:10.654321",
    "interview_active": false,
    "questions_asked": 0,
    "total_expected_questions": 10
  }
}
```

**Example**:
```javascript
// Nest.js implementation
@Get('sessions')
async listSessions() {
  const response = await this.httpService.get(`${API_URL}/sessions`).toPromise();
  return response.data;
}
```

### Delete Session

Manually deletes a session, saving it to a file first.

**Endpoint**: `DELETE /session/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: 
```json
{
  "session_id": "uuid-string",
  "message": "Session saved to path/to/file.json and deleted successfully"
}
```

**Example**:
```javascript
// Nest.js implementation
@Delete('session/:sessionId')
async deleteSession(@Param('sessionId') sessionId: string): Promise<SessionResponse> {
  const response = await this.httpService.delete(`${API_URL}/session/${sessionId}`).toPromise();
  return response.data;
}
```

### End Interview

Explicitly ends an interview by admin command.

**Endpoint**: `POST /end-interview/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**: 
```json
{
  "session_id": "uuid-string",
  "message": "Interview ended successfully"
}
```

**Notes**:
- If there is no active interview in the session, returns a 400 error

**Example**:
```javascript
// Nest.js implementation
@Post('end-interview/:sessionId')
async endInterview(@Param('sessionId') sessionId: string): Promise<SessionResponse> {
  const response = await this.httpService.post(`${API_URL}/end-interview/${sessionId}`).toPromise();
  return response.data;
}
```

### Get Model Info

Returns information about the fixed model being used.

**Endpoint**: `GET /model-info`

**Response**:
```json
{
  "model_name": "llama-3.3-70b-versatile",
  "description": "Pre-configured large language model for interview simulations"
}
```

**Example**:
```javascript
// Nest.js implementation
@Get('model-info')
async getModelInfo() {
  const response = await this.httpService.get(`${API_URL}/model-info`).toPromise();
  return response.data;
}
```

### Get Security Info

Gets information about the interview session security status.

**Endpoint**: `GET /security-info/{session_id}`

**Path Parameters**:
- `session_id`: UUID of the session

**Response**:
```json
{
  "session_id": "uuid-string",
  "potential_cheat_attempts": 0,
  "active": true
}
```

**Example**:
```javascript
// Nest.js implementation
@Get('security-info/:sessionId')
async getSecurityInfo(@Param('sessionId') sessionId: string) {
  const response = await this.httpService.get(`${API_URL}/security-info/${sessionId}`).toPromise();
  return response.data;
}
```

## Interview Flow

The AI interviewer follows this specific flow:

1. **Welcome**: Greets the candidate by name (if provided during session creation)
2. **Personal Questions**: Asks about the candidate's interest in the role
3. **Behavioral Questions**: Asks the specified number of behavioral questions first
4. **Technical Questions**: Asks the specified number of technical questions related to the skills
5. **Custom Questions**: Asks any custom questions provided
6. **Cross-Questioning**: For technical questions, probes deeper with follow-up questions
7. **Closing**: Ends with a professional conclusion and includes "End of interview" marker

## Implementation Notes for Nest.js

1. **HTTP Service Setup**:
   ```typescript
   import { HttpModule } from '@nestjs/axios';
   
   @Module({
     imports: [
       HttpModule,
       // ...other imports
     ],
   })
   export class AppModule {}
   ```

2. **Streaming Responses**:
   - For endpoints that return streamed responses, you'll need to pipe the response directly
   - Use `responseType: 'stream'` option when making HTTP requests

3. **Environment Configuration**:
   - Store the API base URL in environment variables
   - Use Nest.js config module for better management
   ```typescript
   API_URL=process.env.AI_INTERVIEWER_API_URL || 'http://localhost:8000'
   ```

4. **Session Management**:
   - Consider implementing a session cleanup mechanism similar to the original
   - Sessions expire after 30 minutes of inactivity

5. **Error Handling**:
   - Implement proper error handling for API requests
   - Forward appropriate error status codes from the AI Interviewer API

6. **Authentication**:
   - Add authentication middleware before forwarding requests to the AI Interviewer API
   - Consider using JWT or OAuth2 for securing endpoints

## Constraints and Limitations

1. The model name is fixed to "llama-3.3-70b-versatile" and cannot be changed by users
2. The API is designed for streaming responses, which requires specific handling in Nest.js
3. Session timeout is set to 30 minutes of inactivity
4. Background processes are used for session cleanup and saving operations
