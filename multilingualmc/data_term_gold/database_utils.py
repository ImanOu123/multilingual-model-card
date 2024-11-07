import json
import io
import os
import fcntl
from typing import Dict, Any
from tinydb import TinyDB, Query, JSONStorage
from tinydb.middlewares import CachingMiddleware

def reload_db(database_file):
    return TinyDB(database_file, storage=IndentedJSONStorage)

def acquire_lock(database_file):
    lock_file = open(database_file + '.lock', 'w')
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    return lock_file

def release_lock(lock_file):
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()

class IndentedJSONStorage(JSONStorage):
    def write(self, data: Dict[str, Dict[str, Any]]):
        # Move the cursor to the beginning of the file just in case
        self._handle.seek(0)

        # Serialize the database state using the user-provided arguments
        serialized = json.dumps(data, indent=2, **self.kwargs)

        # Write the serialized data to the file
        try:
            self._handle.write(serialized)
        except io.UnsupportedOperation:
            raise IOError('Cannot write to the database. Access mode is "{0}"'.format(self._mode))

        # Ensure the file has been written
        self._handle.flush()
        os.fsync(self._handle.fileno())

        # Remove data that is behind the new cursor in case the file has
        # gotten shorter
        self._handle.truncate()

class TinyDataBase:
    def __init__(self, database_file):
        self.database_file = database_file
        self.db = TinyDB(database_file, storage=IndentedJSONStorage)
    
    def insert(self, profile: dict):
        self.db.insert(profile)

    def build_query(self, **kwargs):
        User = Query()
        query = None
        for key, val in kwargs.items():
            if query is None:
                query = (User[key] == val)
            else:
                query = query & (User[key] == val)
        return query
    
    def get(self, **kwargs):
        return self.db.get(cond=self.build_query(**kwargs))
    
    def get_pk(self, **kwargs):
        res = self.db.get(cond=self.build_query(**kwargs))
        if res is not None:
            return str(res.doc_id)
    
    def get_doc(self, pk: str):
        return self.db.get(doc_id=int(pk))
    
    def search(self, **kwargs):
        return self.db.search(cond=self.build_query(**kwargs))
    
    def safe_search(self, **kwargs):
        lock_file = None
        try:
            lock_file = acquire_lock(self.database_file)
            
            self.db = reload_db(self.database_file)
            return self.db.search(cond=self.build_query(**kwargs))
        finally:
            if lock_file:
                release_lock(lock_file)
    
    def contains(self, **kwargs):
        return self.db.contains(cond=self.build_query(**kwargs))
    
    def safe_contains(self, **kwargs):
        lock_file = None
        try:
            lock_file = acquire_lock(self.database_file)
            
            self.db = reload_db(self.database_file)
            return self.db.contains(cond=self.build_query(**kwargs))
        finally:
            if lock_file:
                release_lock(lock_file)
    
    def remove(self, **kwargs):
        return self.db.remove(cond=self.build_query(**kwargs))
    
    def insert_unique(self, item: dict):
        if self.contains(**item):
            return
        else:
            self.insert(item)
    
    def safe_insert_unique(self, item: dict):
        lock_file = acquire_lock(self.database_file)
        
        self.db = reload_db(self.database_file)
        self.insert_unique(item)
        
        release_lock(lock_file)
    
    def all(self):
        return self.db.all()

    def all_pks(self):
        return [str(doc.doc_id) for doc in self.all()]