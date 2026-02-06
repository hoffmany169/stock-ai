class Node:
    """Node class to store data and link to next node"""
    def __init__(self, data: any):
        self.data = data
        self.next: Node | None = None


class ChainData:
    """Chain data class that maintains a chain of same-type elements"""
    
    def __init__(self):
        """Initialize an empty chain"""
        self.head: Node | None = None
        self._data_type: type | None = None
        self._length: int = 0
    
    def add(self, data: any) -> None:
        """
        Add data to the end of the chain if it matches the chain's type
        
        Args:
            data: Data to add to the chain
            
        Raises:
            TypeError: If data type doesn't match existing chain data type
        """
        if not self._validate_type(data):
            raise TypeError(f"Data must be of type {self._data_type.__name__}, "
                          f"got {type(data).__name__}")
        
        new_node = Node(data)
        
        if self.head is None:
            self.head = new_node
            self._data_type = type(data)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self._length += 1
        return new_node
    
    def insert(self, index: int, data: any) -> None:
        """
        Insert data at a specific position in the chain
        
        Args:
            index: Position to insert (0-based)
            data: Data to insert
            
        Raises:
            TypeError: If data type doesn't match chain data type
            IndexError: If index is out of range
        """
        if not self._validate_type(data):
            raise TypeError(f"Data must be of type {self._data_type.__name__}, "
                          f"got {type(data).__name__}")
        
        if index < 0 or index > self._length:
            raise IndexError(f"Index {index} out of range for chain of length {self._length}")
        
        new_node = Node(data)
        
        # Special case: insert at head
        if index == 0:
            new_node.next = self.head
            self.head = new_node
            if self._data_type is None:
                self._data_type = type(data)
        else:
            current = self.head
            # Move to node before insertion point
            for _ in range(index - 1):
                current = current.next
            new_node.next = current.next
            current.next = new_node
        
        self._length += 1
    
    def get(self, index: int) -> any:
        """
        Get data at a specific index
        
        Args:
            index: Index of data to retrieve (0-based)
            
        Returns:
            The data at the specified index
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range for chain of length {self._length}")
        
        current = self.head
        for _ in range(index):
            current = current.next
        return current.data
    
    def remove(self, index: int) -> any:
        """
        Remove data at a specific index
        
        Args:
            index: Index of data to remove (0-based)
            
        Returns:
            The removed data
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range for chain of length {self._length}")
        
        # Special case: remove head
        if index == 0:
            removed_data = self.head.data
            self.head = self.head.next
        else:
            current = self.head
            # Move to node before the one to remove
            for _ in range(index - 1):
                current = current.next
            removed_data = current.next.data
            current.next = current.next.next
        
        self._length -= 1
        
        # If chain is empty, reset data type
        if self._length == 0:
            self._data_type = None
        
        return removed_data
    
    def clear(self) -> None:
        """Clear all data from the chain"""
        self.head = None
        self._data_type = None
        self._length = 0
    
    def _validate_type(self, data: any) -> bool:
        """
        Validate if data type matches chain's data type
        
        Args:
            data: Data to validate
            
        Returns:
            True if type is valid, False otherwise
        """
        # If chain is empty, any type is allowed
        if self._data_type is None:
            return True
        # Check if data type matches chain's data type
        return isinstance(data, self._data_type)
    
    def __len__(self) -> int:
        """Return the length of the chain"""
        return self._length
    
    def __getitem__(self, index: int) -> any:
        """Enable indexing with chain[index]"""
        return self.get(index)
    
    def __setitem__(self, index: int, data: any) -> None:
        """Enable chain[index] = data"""
        self.remove(index)
        self.insert(index, data)
    
    def __iter__(self):
        """Make the chain iterable"""
        current = self.head
        while current:
            yield current.data
            current = current.next
    
    def __str__(self) -> str:
        """String representation of the chain"""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements) if elements else "Empty Chain"

    def __repr__(self) -> str:
        """Detailed representation of the chain"""
        return f"chain_data(length={self._length}, type={self._data_type.__name__ if self._data_type else 'None'})"
    
    @property
    def data_type(self) -> type|None:
        """Get the data type of the chain"""
        return self._data_type
    
    def contains(self, data: any) -> bool:
        """
        Check if chain contains specific data
        
        Args:
            data: Data to search for
            
        Returns:
            True if data is found, False otherwise
        """
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def index_of(self, data: any) -> int:
        """
        Get index of specific data
        
        Args:
            data: Data to find
            
        Returns:
            Index of data if found, -1 otherwise
        """
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1



if __name__ == "__main__":
    # Create a chain
    chain = ChainData()

    # Add elements (must be same type)
    chain.add(10)
    chain.add(20)
    chain.add(30)
    print("Whole chain: ", chain)

    # Insert at specific position
    chain.insert(1, 15)  # Chain: 10 -> 15 -> 20 -> 30
    print("After insert at index 1 (15): ", chain)
    # Access elements
    print("value with index 2(get): ", chain.get(2))  # 20
    print("value with index 2(indexing): ", chain[2])      # 20 (using indexing)
    # Remove element
    removed = chain.remove(1)  # Removes 15
    print("After remove index 1 (15): ", chain)

    # Clear all
    chain.clear()        

    # Create a string chain
    str_chain = ChainData()
    str_chain.add("hello")
    str_chain.add("world")
    print(f"\nString chain: {str_chain}")
    print(f"\nString chain: {repr(str_chain)}")
    print(f"Contains 'world': {str_chain.contains('world')}")
    print(f"Index of 'world': {str_chain.index_of('world')}")
    # This would raise TypeError:
    str_chain.add(123)  # Can't add integer to string chain