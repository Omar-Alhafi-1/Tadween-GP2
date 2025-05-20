"""
Initialize the database and create a test user
"""
import os
import sys
from datetime import datetime
from langchain_app import create_app, db
from models import User, MinistryContact

app = create_app()

def init_db():
    """Initialize the database and create initial data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database tables created")

        # Create test user if it doesn't exist
        existing_user = User.query.filter_by(username='test').first()
        if not existing_user:
            test_user = User(
                username='test',
                session_id='test-session-id',
                created_at=datetime.utcnow()
            )
            test_user.set_password('admin123')
            db.session.add(test_user)
            print("Created test user (username: test, password: admin123)")

        # Create ministry contacts (initial values)
        if not MinistryContact.query.first():
            contacts = [
                MinistryContact(
                    name="Ministry of Labor - Main Office",
                    position="Main Office",
                    phone="123-456-7890",
                    address="123 Main Street, City",
                    email="main@labor.gov"
                ),
                MinistryContact(
                    name="Ministry of Labor - Branch Office",
                    position="Branch Office",
                    phone="098-765-4321",
                    address="456 Branch Street, City",
                    email="branch@labor.gov"
                )
            ]
            db.session.add_all(contacts)
            print("Created ministry contacts")

        # Commit all changes
        db.session.commit()
        print("Database initialization completed")

if __name__ == '__main__':
    init_db() 