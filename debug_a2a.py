
try:
    from a2a.server.tasks import DatabaseTaskStore
    print(f"Type: {type(DatabaseTaskStore)}")
    print(f"Value: {DatabaseTaskStore}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
