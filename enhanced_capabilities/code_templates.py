"""
Code template library for enhanced code generation capabilities
"""
from typing import Dict, Tuple

# Dictionary of language templates
LANGUAGE_TEMPLATES = {
    # Python Templates
    "python_basic": {
        "description": "Basic Python example",
        "code": """def greet(name):
    \"\"\"Return a greeting message.\"\"\"
    return f"Hello, {name}! Welcome to programming."

# Example usage
user_name = "World"
message = greet(user_name)
print(message)
"""
    },
    "python_flask": {
        "description": "Flask web application",
        "code": """from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data - in a real app, you'd use a database
tasks = [
    {"id": 1, "title": "Study for exams", "completed": False},
    {"id": 2, "title": "Submit assignment", "completed": True}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    return jsonify({"tasks": tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task["id"] == task_id), None)
    if task:
        return jsonify({"task": task})
    return jsonify({"error": "Task not found"}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    if not request.json or 'title' not in request.json:
        return jsonify({"error": "Title is required"}), 400
    
    new_task = {
        "id": tasks[-1]["id"] + 1 if tasks else 1,
        "title": request.json["title"],
        "completed": False
    }
    tasks.append(new_task)
    return jsonify({"task": new_task}), 201

if __name__ == '__main__':
    app.run(debug=True)
"""
    },
    "python_django": {
        "description": "Django model and view",
        "code": """# models.py
from django.db import models

class Course(models.Model):
    title = models.CharField(max_length=200)
    code = models.CharField(max_length=20)
    credits = models.IntegerField(default=3)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.code}: {self.title}"

class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    courses = models.ManyToManyField(Course, related_name='students')
    
    def __str__(self):
        return self.name

# views.py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Course, Student

def course_list(request):
    courses = Course.objects.all()
    return render(request, 'courses/course_list.html', {'courses': courses})

def course_detail(request, course_id):
    course = get_object_or_404(Course, pk=course_id)
    return render(request, 'courses/course_detail.html', {'course': course})

def api_course_list(request):
    courses = Course.objects.all()
    data = [{'id': c.id, 'title': c.title, 'code': c.code} for c in courses]
    return JsonResponse({'courses': data})
"""
    },
    "python_data_science": {
        "description": "Data science with pandas and matplotlib",
        "code": """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create sample student data
np.random.seed(42)
data = {
    'Student_ID': range(1, 101),
    'Hours_Studied': np.random.normal(25, 5, 100),
    'Sleep_Hours': np.random.normal(6, 1, 100),
    'Final_Grade': np.random.normal(75, 10, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate correlation between study hours and grades
correlation = df['Hours_Studied'].corr(df['Final_Grade'])
print(f"Correlation between study hours and grades: {correlation:.2f}")

# Basic statistics
print("\\nStatistical Summary:")
print(df.describe())

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Hours_Studied'], df['Final_Grade'], alpha=0.6)
plt.title('Relationship Between Study Hours and Final Grades')
plt.xlabel('Hours Studied')
plt.ylabel('Final Grade (%)')

# Add regression line
z = np.polyfit(df['Hours_Studied'], df['Final_Grade'], 1)
p = np.poly1d(z)
plt.plot(df['Hours_Studied'], p(df['Hours_Studied']), "r--")

plt.grid(True, alpha=0.3)
plt.savefig('study_grade_relationship.png')
plt.show()

# Group by grade ranges
df['Grade_Range'] = pd.cut(df['Final_Grade'], 
                           bins=[0, 60, 70, 80, 90, 100],
                           labels=['F', 'C', 'B', 'A', 'A+'])

grade_counts = df['Grade_Range'].value_counts().sort_index()
print("\\nGrade Distribution:")
print(grade_counts)

# Bar chart of grade distribution
plt.figure(figsize=(8, 5))
grade_counts.plot(kind='bar', color='skyblue')
plt.title('Grade Distribution')
plt.xlabel('Grade Range')
plt.ylabel('Number of Students')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('grade_distribution.png')
plt.show()
"""
    },

    # JavaScript Templates
    "javascript_basic": {
        "description": "Basic JavaScript example",
        "code": """// A simple JavaScript function
function calculateGrade(score) {
    if (score >= 90) {
        return 'A';
    } else if (score >= 80) {
        return 'B';
    } else if (score >= 70) {
        return 'C';
    } else if (score >= 60) {
        return 'D';
    } else {
        return 'F';
    }
}

// Example usage
const studentScores = [95, 82, 78, 65, 59];
const studentNames = ['Alice', 'Bob', 'Charlie', 'David', 'Eve'];

// Process each student's score
for (let i = 0; i < studentScores.length; i++) {
    const grade = calculateGrade(studentScores[i]);
    console.log(`${studentNames[i]}'s grade: ${grade}`);
}

// Calculate class average
const average = studentScores.reduce((sum, score) => sum + score, 0) / studentScores.length;
console.log(`Class average: ${average.toFixed(1)}`);
"""
    },
    "javascript_react": {
        "description": "React component example",
        "code": """import React, { useState, useEffect } from 'react';

// A simple Task component
const Task = ({ task, onComplete, onDelete }) => {
  return (
    <div className="task" style={{ 
      padding: '10px', 
      margin: '10px 0',
      borderLeft: '4px solid #8B5CF6',
      backgroundColor: task.completed ? '#f0f0f0' : 'white',
      display: 'flex',
      justifyContent: 'space-between'
    }}>
      <div>
        <h3 style={{ 
          textDecoration: task.completed ? 'line-through' : 'none',
          color: task.completed ? '#888' : '#333' 
        }}>
          {task.title}
        </h3>
        <p>{task.description}</p>
      </div>
      <div>
        <button 
          onClick={() => onComplete(task.id)}
          style={{
            marginRight: '5px',
            backgroundColor: task.completed ? '#ccc' : '#8B5CF6',
            color: 'white',
            border: 'none',
            padding: '5px 10px',
            borderRadius: '4px'
          }}
        >
          {task.completed ? 'Undo' : 'Complete'}
        </button>
        <button 
          onClick={() => onDelete(task.id)}
          style={{
            backgroundColor: '#ff4d4f',
            color: 'white',
            border: 'none',
            padding: '5px 10px',
            borderRadius: '4px'
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
};

// Task List component with state management
function TaskList() {
  const [tasks, setTasks] = useState([]);
  const [newTask, setNewTask] = useState({ title: '', description: '' });
  const [isLoading, setIsLoading] = useState(true);

  // Simulate fetching tasks from an API
  useEffect(() => {
    const fetchTasks = async () => {
      // In a real app, you'd fetch from an API
      setTimeout(() => {
        setTasks([
          { id: 1, title: 'Complete Assignment', description: 'Finish the React project', completed: false },
          { id: 2, title: 'Study for Exam', description: 'Review chapter 7 and 8', completed: false },
          { id: 3, title: 'Team Meeting', description: 'Discuss project progress', completed: true }
        ]);
        setIsLoading(false);
      }, 1000);
    };
    
    fetchTasks();
  }, []);

  const handleAddTask = (e) => {
    e.preventDefault();
    if (!newTask.title) return;
    
    const task = {
      id: Date.now(),
      title: newTask.title,
      description: newTask.description,
      completed: false
    };
    
    setTasks([...tasks, task]);
    setNewTask({ title: '', description: '' });
  };

  const toggleComplete = (taskId) => {
    setTasks(tasks.map(task => 
      task.id === taskId ? { ...task, completed: !task.completed } : task
    ));
  };

  const deleteTask = (taskId) => {
    setTasks(tasks.filter(task => task.id !== taskId));
  };

  if (isLoading) {
    return <div>Loading tasks...</div>;
  }

  return (
    <div className="task-list">
      <h2>Student Task Manager</h2>
      
      <form onSubmit={handleAddTask} style={{ marginBottom: '20px' }}>
        <input
          type="text"
          placeholder="Task title"
          value={newTask.title}
          onChange={(e) => setNewTask({...newTask, title: e.target.value})}
          style={{ padding: '8px', marginRight: '10px', width: '200px' }}
        />
        <input
          type="text"
          placeholder="Description"
          value={newTask.description}
          onChange={(e) => setNewTask({...newTask, description: e.target.value})}
          style={{ padding: '8px', marginRight: '10px', width: '300px' }}
        />
        <button 
          type="submit"
          style={{
            backgroundColor: '#8B5CF6',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '4px'
          }}
        >
          Add Task
        </button>
      </form>
      
      <div>
        {tasks.length === 0 ? (
          <p>No tasks yet. Add a task to get started!</p>
        ) : (
          tasks.map(task => (
            <Task 
              key={task.id} 
              task={task} 
              onComplete={toggleComplete} 
              onDelete={deleteTask} 
            />
          ))
        )}
      </div>
      
      <div style={{ marginTop: '20px' }}>
        <p><strong>Statistics:</strong></p>
        <p>Total Tasks: {tasks.length}</p>
        <p>Completed: {tasks.filter(t => t.completed).length}</p>
        <p>Pending: {tasks.filter(t => !t.completed).length}</p>
      </div>
    </div>
  );
}

export default TaskList;
"""
    },
    "javascript_node": {
        "description": "Node.js Express API",
        "code": """const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// In-memory database for demonstration
let courses = [
  { id: 1, code: 'CS101', title: 'Introduction to Computer Science', credits: 3 },
  { id: 2, code: 'MATH201', title: 'Calculus I', credits: 4 },
  { id: 3, code: 'ENG105', title: 'Academic Writing', credits: 3 }
];

// Middleware to log requests
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Get all courses
app.get('/api/courses', (req, res) => {
  res.json({ courses });
});

// Get a specific course
app.get('/api/courses/:id', (req, res) => {
  const courseId = parseInt(req.params.id);
  const course = courses.find(c => c.id === courseId);
  
  if (!course) {
    return res.status(404).json({ error: 'Course not found' });
  }
  
  res.json({ course });
});

// Create a new course
app.post('/api/courses', (req, res) => {
  const { code, title, credits } = req.body;
  
  // Validate request
  if (!code || !title || !credits) {
    return res.status(400).json({ error: 'Code, title, and credits are required' });
  }
  
  // Create new course object
  const newCourse = {
    id: courses.length > 0 ? Math.max(...courses.map(c => c.id)) + 1 : 1,
    code,
    title,
    credits: parseInt(credits)
  };
  
  courses.push(newCourse);
  res.status(201).json({ course: newCourse });
});

// Update a course
app.put('/api/courses/:id', (req, res) => {
  const courseId = parseInt(req.params.id);
  const courseIndex = courses.findIndex(c => c.id === courseId);
  
  if (courseIndex === -1) {
    return res.status(404).json({ error: 'Course not found' });
  }
  
  const { code, title, credits } = req.body;
  
  // Update course with new data, keeping existing values if not provided
  courses[courseIndex] = {
    ...courses[courseIndex],
    code: code || courses[courseIndex].code,
    title: title || courses[courseIndex].title,
    credits: credits !== undefined ? parseInt(credits) : courses[courseIndex].credits
  };
  
  res.json({ course: courses[courseIndex] });
});

// Delete a course
app.delete('/api/courses/:id', (req, res) => {
  const courseId = parseInt(req.params.id);
  const initialLength = courses.length;
  
  courses = courses.filter(c => c.id !== courseId);
  
  if (courses.length === initialLength) {
    return res.status(404).json({ error: 'Course not found' });
  }
  
  res.json({ message: 'Course deleted successfully' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
"""
    },

    # HTML/CSS Templates
    "html_basic": {
        "description": "Basic HTML template",
        "code": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        header {
            background-color: #f4f4f4;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Welcome to My Website</h1>
        </header>
        <main>
            <p>This is a simple HTML template to get you started.</p>
        </main>
        <footer>
            <p>&copy; 2025 My Website. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>"""
    },
    "html_portfolio": {
        "description": "Portfolio website template",
        "code": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Portfolio | ALU</title>
    <style>
        :root {
            --primary: #8B5CF6;
            --primary-light: #9b87f5;
            --dark: #1A1F2C;
            --dark-light: #2A2F3C;
            --light: #f5f5f7;
            --gray: #6b7280;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: var(--light);
        }
        
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header styles */
        header {
            background: linear-gradient(to right, var(--primary-light), var(--primary));
            color: white;
            padding: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
        }
        
        .nav-links li {
            margin-left: 20px;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: opacity 0.3s ease;
        }
        
        .nav-links a:hover {
            opacity: 0.8;
        }
        
        /* Hero section */
        .hero {
            padding: 80px 0;
            background-color: var(--dark);
            color: white;
            text-align: center;
        }
        
        .hero h1 {
            font-size: 3rem;
            margin-bottom: 20px;
        }
        
        .hero p {
            font-size: 1.2rem;
            max-width: 700px;
            margin: 0 auto;
            color: var(--gray);
        }
        
        .cta-button {
            display: inline-block;
            background: linear-gradient(to right, var(--primary-light), var(--primary));
            color: white;
            padding: 12px 30px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 30px;
            transition: transform 0.3s ease;
        }
        
        .cta-button:hover {
            transform: translateY(-5px);
        }
        
        /* About section */
        .section {
            padding: 80px 0;
        }
        
        .section-title {
            text-align: center;
            margin-bottom: 50px;
            font-size: 2rem;
            position: relative;
        }
        
        .section-title::after {
            content: '';
            display: block;
            width: 50px;
            height: 4px;
            background-color: var(--primary);
            margin: 15px auto 0;
        }
        
        .about-content {
            display: flex;
            align-items: center;
            gap: 40px;
        }
        
        .about-image {
            flex: 1;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .about-image img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .about-text {
            flex: 1;
        }
        
        .about-text h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: var(--primary);
        }
        
        /* Projects section */
        .projects {
            background-color: #f9fafb;
        }
        
        .projects-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 30px;
        }
        
        .project-card {
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .project-card:hover {
            transform: translateY(-10px);
        }
        
        .project-image {
            width: 100%;
            height: 200px;
            background-color: var(--dark-light);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
        }
        
        .project-content {
            padding: 20px;
        }
        
        .project-content h3 {
            margin-bottom: 10px;
            color: var(--dark);
        }
        
        .project-content p {
            color: var(--gray);
            margin-bottom: 15px;
        }
        
        .project-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        
        .tag {
            background-color: #e5e7eb;
            color: var(--gray);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        
        /* Footer */
        footer {
            background-color: var(--dark);
            color: white;
            padding: 40px 0;
            text-align: center;
        }
        
        .social-links {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            gap: 15px;
        }
        
        .social-link {
            display: inline-block;
            width: 40px;
            height: 40px;
            background-color: var(--dark-light);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            text-decoration: none;
            transition: background-color 0.3s ease;
        }
        
        .social-link:hover {
            background-color: var(--primary);
        }
        
        @media (max-width: 768px) {
            .about-content {
                flex-direction: column;
            }
            
            .hero h1 {
                font-size: 2rem;
            }
            
            .projects-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">StudentName</div>
                <ul class="nav-links">
                    <li><a href="#about">About</a></li>
                    <li><a href="#projects">Projects</a></li>
                    <li><a href="#skills">Skills</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="hero">
        <div class="container">
            <h1>Student Name</h1>
            <p>Computer Science Student | African Leadership University</p>
            <a href="#contact" class="cta-button">Get in Touch</a>
        </div>
    </section>
    
    <section id="about" class="section">
        <div class="container">
            <h2 class="section-title">About Me</h2>
            <div class="about-content">
                <div class="about-image">
                    <!-- Placeholder for profile image -->
                    <div style="background-color: #ddd; height: 300px; display: flex; align-items: center; justify-content: center;">
                        Profile Image
                    </div>
                </div>
                <div class="about-text">
                    <h3>Computer Science Student at ALU</h3>
                    <p>
                        I am a passionate computer science student at African Leadership University with a focus on 
                        software development and artificial intelligence. My goal is to create technology solutions
                        that address unique African challenges.
                    </p>
                    <p>
                        During my time at ALU, I've developed a strong foundation in programming, algorithms, data structures,
                        and software engineering principles. I'm particularly interested in web development, machine learning,
                        and building accessible applications.
                    </p>
                </div>
            </div>
        </div>
    </section>
    
    <section id="projects" class="section projects">
        <div class="container">
            <h2 class="section-title">My Projects</h2>
            <div class="projects-grid">
                <!-- Project 1 -->
                <div class="project-card">
                    <div class="project-image">Project 1</div>
                    <div class="project-content">
                        <h3>Student Course Management System</h3>
                        <p>A web application to help students track their courses, assignments, and academic progress.</p>
                        <div class="project-tags">
                            <span class="tag">React</span>
                            <span class="tag">Node.js</span>
                            <span class="tag">MongoDB</span>
                        </div>
                    </div>
                </div>
                
                <!-- Project 2 -->
                <div class="project-card">
                    <div class="project-image">Project 2</div>
                    <div class="project-content">
                        <h3>Community Health Tracker</h3>
                        <p>Mobile app for tracking health metrics and accessing local healthcare resources.</p>
                        <div class="project-tags">
                            <span class="tag">Flutter</span>
                            <span class="tag">Firebase</span>
                            <span class="tag">RESTful API</span>
                        </div>
                    </div>
                </div>
                
                <!-- Project 3 -->
                <div class="project-card">
                    <div class="project-image">Project 3</div>
                    <div class="project-content">
                        <h3>Agricultural Market Analysis</h3>
                        <p>Data analysis project examining agricultural market trends across East Africa.</p>
                        <div class="project-tags">
                            <span class="tag">Python</span>
                            <span class="tag">Pandas</span>
                            <span class="tag">Data Visualization</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <footer>
        <div class="container">
            <h3>Get in Touch</h3>
            <div class="social-links">
                <a href="#" class="social-link">LI</a>
                <a href="#" class="social-link">GH</a>
                <a href="#" class="social-link">TW</a>
            </div>
            <p>Email: student@alustudent.com</p>
            <p>&copy; 2025 Student Portfolio. African Leadership University.</p>
        </div>
    </footer>
</body>
</html>"""
    },

    # SQL Templates
    "sql_database": {
        "description": "SQL Database schema for a university",
        "code": """-- University Database Schema

-- Create Students table
CREATE TABLE Students (
    student_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    date_of_birth DATE,
    enrollment_date DATE NOT NULL,
    graduation_date DATE,
    major_id INT,
    gpa DECIMAL(3,2),
    FOREIGN KEY (major_id) REFERENCES Majors(major_id)
);

-- Create Professors table
CREATE TABLE Professors (
    professor_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    department_id INT,
    hire_date DATE NOT NULL,
    title VARCHAR(50),
    FOREIGN KEY (department_id) REFERENCES Departments(department_id)
);

-- Create Departments table
CREATE TABLE Departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    building VARCHAR(50),
    budget DECIMAL(12,2),
    chair_id INT,
    FOREIGN KEY (chair_id) REFERENCES Professors(professor_id)
);

-- Create Majors table
CREATE TABLE Majors (
    major_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    department_id INT,
    required_credits INT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id)
);

-- Create Courses table
CREATE TABLE Courses (
    course_id INT PRIMARY KEY AUTO_INCREMENT,
    code VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    credits INT NOT NULL,
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id)
);

-- Create Course Offerings table
CREATE TABLE CourseOfferings (
    offering_id INT PRIMARY KEY AUTO_INCREMENT,
    course_id INT NOT NULL,
    professor_id INT,
    semester VARCHAR(20) NOT NULL,
    year INT NOT NULL,
    start_date DATE,
    end_date DATE,
    room VARCHAR(50),
    schedule VARCHAR(100),
    max_students INT,
    FOREIGN KEY (course_id) REFERENCES Courses(course_id),
    FOREIGN KEY (professor_id) REFERENCES Professors(professor_id)
);

-- Create Enrollments table
CREATE TABLE Enrollments (
    enrollment_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    offering_id INT NOT NULL,
    enrollment_date DATE NOT NULL,
    grade CHAR(2),
    completion_status VARCHAR(20) DEFAULT 'In Progress',
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (offering_id) REFERENCES CourseOfferings(offering_id)
);

-- Create Assignments table
CREATE TABLE Assignments (
    assignment_id INT PRIMARY KEY AUTO_INCREMENT,
    offering_id INT NOT NULL,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    due_date DATETIME,
    max_points INT,
    weight DECIMAL(5,2),
    FOREIGN KEY (offering_id) REFERENCES CourseOfferings(offering_id)
);

-- Create Assignment Submissions table
CREATE TABLE AssignmentSubmissions (
    submission_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT NOT NULL,
    student_id INT NOT NULL,
    submission_date DATETIME,
    file_path VARCHAR(255),
    comments TEXT,
    grade DECIMAL(5,2),
    graded_by INT,
    FOREIGN KEY (assignment_id) REFERENCES Assignments(assignment_id),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (graded_by) REFERENCES Professors(professor_id)
);

-- Sample queries

-- Get all courses for a specific student in the current semester
SELECT c.code, c.title, co.schedule, p.first_name, p.last_name
FROM Courses c
JOIN CourseOfferings co ON c.course_id = co.course_id
JOIN Enrollments e ON co.offering_id = e.offering_id
JOIN Students s ON e.student_id = s.student_id
JOIN Professors p ON co.professor_id = p.professor_id
WHERE s.student_id = 1 
  AND co.semester = 'Spring'
  AND co.year = 2025;

-- Calculate GPA for a student
SELECT s.student_id, s.first_name, s.last_name,
       AVG(CASE
           WHEN e.grade = 'A+' THEN 4.0
           WHEN e.grade = 'A' THEN 4.0
           WHEN e.grade = 'A-' THEN 3.7
           WHEN e.grade = 'B+' THEN 3.3
           WHEN e.grade = 'B' THEN 3.0
           WHEN e.grade = 'B-' THEN 2.7
           WHEN e.grade = 'C+' THEN 2.3
           WHEN e.grade = 'C' THEN 2.0
           WHEN e.grade = 'C-' THEN 1.7
           WHEN e.grade = 'D+' THEN 1.3
           WHEN e.grade = 'D' THEN 1.0
           WHEN e.grade = 'F' THEN 0.0
       END) AS gpa
FROM Students s
JOIN Enrollments e ON s.student_id = e.student_id
JOIN CourseOfferings co ON e.offering_id = co.offering_id
JOIN Courses c ON co.course_id = c.course_id
WHERE s.student_id = 1
  AND e.grade IS NOT NULL;

-- Find professors with the most courses taught
SELECT p.professor_id, p.first_name, p.last_name, COUNT(co.offering_id) AS courses_taught
FROM Professors p
JOIN CourseOfferings co ON p.professor_id = co.professor_id
GROUP BY p.professor_id
ORDER BY courses_taught DESC
LIMIT 5;"""
    },

    # More languages and frameworks can be added here
}

def get_template(language_key: str) -> Tuple[str, str]:
    """Get a code template by its key."""
    if language_key in LANGUAGE_TEMPLATES:
        template = LANGUAGE_TEMPLATES[language_key]
        return template["description"], template["code"]
    else:
        # Return a basic template as fallback
        return "Basic code example", "// No specific template found for this language"

def get_all_template_keys() -> list:
    """Get a list of all available template keys."""
    return list(LANGUAGE_TEMPLATES.keys())

def guess_template_from_request(request: str) -> str:
    """Guess the most appropriate template based on the user's request."""
    request = request.lower()
    
    # Map common keywords to templates
    keyword_map = {
        "flask": "python_flask",
        "django": "python_django",
        "data": "python_data_science", 
        "pandas": "python_data_science",
        "matplotlib": "python_data_science",
        "visualization": "python_data_science",
        "react": "javascript_react",
        "component": "javascript_react",
        "node": "javascript_node",
        "express": "javascript_node",
        "api": "javascript_node",
        "portfolio": "html_portfolio",
        "website": "html_portfolio",
        "sql": "sql_database",
        "database": "sql_database",
        "schema": "sql_database"
    }
    
    # Check if any specific framework/library is mentioned
    for keyword, template in keyword_map.items():
        if keyword in request:
            return template
    
    # Fall back to language-specific templates
    language_map = {
        "python": "python_basic",
        "javascript": "javascript_basic",
        "js": "javascript_basic",
        "html": "html_basic",
        "css": "html_basic",
        "sql": "sql_database"
    }
    
    for lang, template in language_map.items():
        if lang in request:
            return template
    
    # Default to Python if no specific match
    return "python_basic"