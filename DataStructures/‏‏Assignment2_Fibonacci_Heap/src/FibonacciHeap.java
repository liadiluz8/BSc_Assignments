/**
 * 
 * Student 1:
 * Name: Guy Levy
 * ID: 318645694
 * Username: gl2
 * 
 * Student 2:
 * Name: Liad Iluz
 * ID: 208427435
 * Username: liadiluz
 * 
 */





/**
 * FibonacciHeap
 *
 * An implementation of a Fibonacci Heap over integers.
 */
public class FibonacciHeap
{
	private int size;
	private HeapNode first;
	private HeapNode min;
	private static int totalLinkes;
	private static int totalCuts;
	private int numOfMarked;
	private int numOfTrees;
	
	
   /**
    * public boolean isEmpty()
    *
    * Returns true if and only if the heap is empty.
    * 
    * Complexity: O(1).
    */
    public boolean isEmpty()
    {
    	return this.size == 0;
    }
		

   /**
    * public HeapNode insert(int key)
    *
    * Creates a node (of type HeapNode) which contains the given key, and inserts it into the heap.
    * The added key is assumed not to already belong to the heap.  
    * 
    * Returns the newly created node.
    * 
    * Complexity: O(1).
    */
    public HeapNode insert(int key)
    {    
    	HeapNode newNode = new HeapNode(key);
        
        if (!this.isEmpty()){ 				// updating pointers for previous first and last.
            HeapNode tmpPointerFirst = this.first;
            HeapNode tmpPointerLast = this.first.getPrev();
            
            newNode.setNext(tmpPointerFirst);
            newNode.setPrev(tmpPointerLast);
            tmpPointerFirst.setPrev(newNode);
            tmpPointerLast.setNext(newNode);        
        }
        
        this.first = newNode; 				// update first
        
        if (this.min == null){ 				// if there was not a min - update min
            this.min = newNode;
        }
        else{
            if (key < this.min.getKey()){ 	// if there is a min - update min if needed
                this.min = newNode;
            }
        }
        
        this.size++; 		// update size
        this.numOfTrees++; 	// update numOfTrees
        
        return newNode;
    }


   /**
    * public void deleteMin()
    *
    * Deletes the node containing the minimum key.
    *
    * Complexity WC: O(n), Amortized: O(log n).
    */
    public void deleteMin()
    {
        if (this.isEmpty()){
            return;
        }
        
        if (this.size == 1){ 		// if min is the only node in the heap
            this.size = 0;
            this.first = null;
            this.min = null;
            this.numOfTrees = 0;
            return;
        }
        
        HeapNode minNode = this.min;
        cutChildrenDeleteMin(minNode); 	// cutting min's children and inserting after min (not in the beginning). Complexity: O(log n).
        
        if (this.first == minNode){ 	// update this.first if needed
            this.first = minNode.getNext();
        }

        minNode.getPrev().setNext(minNode.getNext());
        minNode.getNext().setPrev(minNode.getPrev());
        minNode.setNext(minNode);
        minNode.setPrev(minNode);

        this.numOfTrees--;
        this.size--; 			// update size of heap
        this.updateMin(); 		// update minimum. // Complexity WC: O(n), Amortized: O(log n).
        this.successiveLink();	// Complexity WC: O(n), Amortized: O(log n).
    }


   /**
    * public HeapNode findMin()
    *
    * Returns the node of the heap whose key is minimal, or null if the heap is empty.
    *
    *Complexity: O(1).
    */
    public HeapNode findMin()
    {
    	return this.min;
    } 
    

   /**
    * public void meld (FibonacciHeap heap2)
    *
    * Melds heap2 with the current heap.
    *
    * Complexity: O(1).
    */
    public void meld (FibonacciHeap heap2)
    {   
        if(heap2.isEmpty()){ 	// if heap2 is empty -> do nothing
            return;
        }
        if (this.isEmpty()){	// if this heap is empty and heap2 isn't empty -> adopt heap2 properties.
            this.size = heap2.size;
            this.first = heap2.first;
            this.min = heap2.min;
            this.numOfMarked = heap2.numOfMarked;
            this.numOfTrees = heap2.numOfTrees;
            return;
        }
        
        // both heaps are not empty - concatenate
        HeapNode tmpPointerFirst1 = this.first;
        HeapNode tmpPointerLast1 = this.first.getPrev();
        HeapNode tmpPointerFirst2 = heap2.first;
        HeapNode tmpPointerLast2 = heap2.first.getPrev();
            
        tmpPointerFirst1.setPrev(tmpPointerLast2);
        tmpPointerLast2.setNext(tmpPointerFirst1);
        tmpPointerLast1.setNext(tmpPointerFirst2);
        tmpPointerFirst2.setPrev(tmpPointerLast1);
            
        this.size += heap2.size; // update relevant fields.
        if (heap2.min.getKey() < this.min.getKey()){
            this.min = heap2.min;
        }
        this.numOfMarked += heap2.numOfMarked;
        this.numOfTrees += heap2.numOfTrees;
    }


   /**
    * public int size()
    *
    * Returns the number of elements in the heap.
    * 
    * Complexity: O(1).
    */
    public int size()
    {
    	return this.size;
    }
    	

    /**
    * public int[] countersRep()
    *
    * Return an array of counters. The i-th entry contains the number of trees of order i in the heap.
    * Note: The size of of the array depends on the maximum order of a tree, and an empty heap returns an empty array.
    * 
    * Complexity: O(this.numOfTrees) <= O(n).
    */
    public int[] countersRep()
    {
    	if (this.isEmpty()) {	
    		return new int[0];
    	}
    	
    	int maxOrder = this.getMaxTreeOrder();	// O(#Trees_in_heap)
    	
    	int[] arr = new int[maxOrder + 1];  	// init with zeros
    	
    	HeapNode curr = this.first;
    	int firstKey = this.first.getKey();
    	arr[curr.getRank()]++;
    	
    	while(curr.getNext().getKey() != firstKey) {	// O(#Trees_in_heap)
			curr = curr.getNext();
			arr[curr.getRank()]++;
    	}
    	
    	return arr;
    }
	
    
   /**
    * public void delete(HeapNode x)
    *
    * Deletes the node x from the heap.
	* It is assumed that x indeed belongs to the heap.
    *
    * Complexity: O(log n).
    */
    public void delete(HeapNode x) 
    {    
    	decreaseKey(x, x.getKey() - this.min.getKey() + 1);  // Complexity WC: O(log n), Amortized: O(1).
    	deleteMin();						// Complexity: O(log n).
    }


   /**
    * public void decreaseKey(HeapNode x, int delta)
    *
    * Decreases the key of the node x by a non-negative value delta. The structure of the heap should be updated
    * to reflect this change (for example, the cascading cuts procedure should be applied if needed).
    * 
    * Complexity WC: O(log n), Amortized: O(1).
    */
    public void decreaseKey(HeapNode x, int delta)
    {    
    	x.setKey(x.getKey() - delta);
    	
    	if (x.getParent() != null && x.getParent().getKey() > x.getKey()) {	// If the rule of the tree violated with its parent after decreasing x's key.
    		cascading_cut(x, x.getParent());	// x.getParent() != null. Complexity WC: O(log n). Amortized: O(1).
    	}
    	
    	//Update the minimum after decreasing x's key. 
    	if (x.getKey() < this.min.key) {
    		this.min = x;
    	} 
    }
    
    
   /**
    * public int potential() 
    *
    * This function returns the current potential of the heap, which is:
    * Potential = #trees + 2*#marked
    * 
    * In words: The potential equals to the number of trees in the heap
    * plus twice the number of marked nodes in the heap. 
    * 
    * Complexity: O(1).
    */
    public int potential() 
    {    
    	return this.numOfTrees + 2*this.numOfMarked;
    }


   /**
    * public static int totalLinks() 
    *
    * This static function returns the total number of link operations made during the
    * run-time of the program. A link operation is the operation which gets as input two
    * trees of the same rank, and generates a tree of rank bigger by one, by hanging the
    * tree which has larger value in its root under the other tree.
    * 
    * Complexity: O(1).
    */
    public static int totalLinks()
    {    
    	return FibonacciHeap.totalLinkes;
    }


   /**
    * public static int totalCuts() 
    *
    * This static function returns the total number of cut operations made during the
    * run-time of the program. A cut operation is the operation which disconnects a subtree
    * from its parent (during decreaseKey/delete methods).
    * 
    * Complexity: O(1).
    */
    public static int totalCuts()
    {    
    	return FibonacciHeap.totalCuts;
    }


     /**
    * public static int[] kMin(FibonacciHeap H, int k) 
    *
    * This static function returns the k smallest elements in a Fibonacci heap that contains a single tree.
    * The function should run in O(k*deg(H)). (deg(H) is the degree of the only tree in H.)
    *  
    * ###CRITICAL### : you are NOT allowed to change H. 
    * 
    * Complexity: O(k*deg(H)).
    */
    public static int[] kMin(FibonacciHeap H, int k)
    {    
        if (k == 0 || H.isEmpty()){
            return new int[0];
        }
        
        FibonacciHeap h2 = new FibonacciHeap();		// Temporery heap for retrieving the minimal node every time.
        int[] resArr = new int[k];
        int i = 1;
        boolean flag = true;

        HeapNode currMin = H.findMin();
        HeapNode currChild = currMin.getChild();
        HeapNode lastChild;
        if (currChild != null){
            lastChild = currChild.getPrev();
        }
        else{
            lastChild = null;
            flag = false; 
        }
        
        resArr[0] = currMin.getKey(); // insert min.key as first element of the array

        while (i < k){ // until we find enough elements
            while (flag){ // insert children to heap
                if (currChild == lastChild){
                    flag = false;
                }
                h2.insertNodePointer(currChild);
                currChild = currChild.getNext();
            }
            flag = true;
            if (!h2.isEmpty()){ // retrieve minimum of h2 and continue from its origin node in the original heap
                currMin = h2.findMin().getOrigin();
                h2.deleteMin();
            }
            resArr[i] = currMin.getKey();
            currChild = currMin.getChild();
            if (currChild != null){
                lastChild = currChild.getPrev();
            }
            else{
                lastChild = null;
                flag = false;
            }
            i++;
        }

        return resArr;
    }


    // Utility Functions


    public HeapNode getFirst() {
		return this.first;
	}


    /**
     * private int getMaxTreeOrder()
     * 
     * @pre: this.isEmpty() == false
     * @return: The max order (rank = number of children) of a tree in the heap
     * 
     * Complexity: O(this.numOfTrees) <= O(n).
     */
    private int getMaxTreeOrder() {	
    	int maxOrder = 0;
    	
    	HeapNode curr = this.first;
     	int firstKey = this.first.getKey();
    	maxOrder = this.first.getRank();
    	
    	while(curr.getNext().getKey() != firstKey) {	// O(this.numOfTrees).
    		curr = curr.getNext();
    		if (curr.getRank() > maxOrder) {
    			maxOrder = curr.getRank();
    		}
    	}
    	
    	return maxOrder;
    }


    /**
     * private void updateMin()
     * 
     * The function updates the min HeapNode in this heap.
     * 
     * Complexity: O(this.numOfTrees) <= O(n).
     */
    private void updateMin() {
    	HeapNode curr = this.first;
    	int firstKey = this.first.getKey();
    	int minKey = this.first.getKey();
    	this.min = curr;

    	while(curr.getNext().getKey() != firstKey) {	// O(this.numOfTrees).
    		curr = curr.getNext();
    		if (curr.getKey() < minKey) {
    			minKey = curr.getKey();
    			this.min = curr;
    		}
    	}
    }

    
    /**
     * private void cascading_cut(HeapNode x, HeapNode y)
     * 
     * Perform a cascading cut process starting at x.
     * @param x
     * @param y == x.getParent().
     * @pre x,y != null.
     * 
     * Complexity WC: O(log n), Amortized: O(1).
     */
    private void cascading_cut(HeapNode x, HeapNode y) {
    	cut(x,y);	// Complexity: O(1).
    	
    	if (y.getParent() != null) {
    		if (y.isMarked()) {
    			cascading_cut(y, y.getParent());	// Recursive call. Complexity WC: O(log n)
    		}
    		else {
    			y.setMark(true);
    			this.numOfMarked++;
    		}
    	}
    }
     

    /**
     * private void cut(HeapNode x, HeapNode y)
     * 
     * Cut x from its parent y and insert it back to the heap
     * @param x
     * @param y
     * @pre x,y != null.
     * 
     * Complexity: O(1).
     */
    private void cut(HeapNode x, HeapNode y) {
    	FibonacciHeap.totalCuts++;
    	this.numOfTrees++;
    	
    	x.setParent(null);
    	if (x.isMarked()) {		// If x was marked, Now x need to be unmarked as a root of new tree.
            x.setMark(false);
    		this.numOfMarked--;
    	}
    	y.setRank(y.getRank() - 1);
    	
    	// Treating the children of y after cut x.
    	if (x.getNext() == x) {			// If x was the only child of y
    		y.setChild(null);
    	}
    	else {
    		if (y.getChild() == x) {	// If x was the child of y and there are more children for y				
    			y.setChild(x.getNext());
    		}
            x.getPrev().setNext(x.getNext());
            x.getNext().setPrev(x.getPrev());
    	}
    	
    	// Link x to the heap as a new tree in the most left tree (as a new first >> (post)this.first == x)
    	HeapNode oldFirst = this.first;
    	this.first = x;
    	x.setNext(oldFirst);
    	x.setPrev(oldFirst.getPrev());
    	oldFirst.getPrev().setNext(x);
    	oldFirst.setPrev(x);
    }


    /**
     * private void successiveLink()
     * 
     * Complexity WC: O(n), Amortized: O(log n).
     */
    private void successiveLink(){
        int logOfn = (int)(Math.log(this.size) / Math.log(2));
        HeapNode[] bucketsArr = new HeapNode[logOfn + 2]; // create buckets array with the size of 1+ the largest degree possible in the heap.
        HeapNode curr = this.first;
        HeapNode nextCurr = curr;
        HeapNode last = this.first.getPrev();
        boolean flag = true;

        while (flag){ // go through each root in the heap.
            if (curr == last){
                flag = false;
            }

            int currRank = curr.getRank();
            if (bucketsArr[currRank] == null){ // if the bucket of its rank is empty, disconnect it from the heap and put it in the bucket.
                bucketsArr[currRank] = curr;
                nextCurr = curr.getNext();
                curr.getPrev().setNext(curr.getNext()); 
                curr.getNext().setPrev(curr.getPrev()); 
                curr.setNext(curr); 
                curr.setPrev(curr); 
                curr = nextCurr;
            }
            else{ // if the bucket is not empty, link successively until you reach an empty bucket.
                nextCurr = curr.getNext();
                while (bucketsArr[currRank] != null){
                    curr = linkTrees(bucketsArr[currRank], curr);
                    bucketsArr[currRank] = null;
                    currRank = curr.getRank();
                }
                bucketsArr[currRank] = curr;
                curr = nextCurr;
            }
        }

        HeapNode[] fullBucketsArr = new HeapNode[this.numOfTrees]; // new array of roots in non empty buckets, in order.
        int i = 0;
        for (HeapNode root : bucketsArr){
            if (root != null){
                fullBucketsArr[i] = root;
                i++;
            }
        }

        for (int j = 0; j < fullBucketsArr.length - 1; j++){ // connecting the roots in order to form the heap.
            fullBucketsArr[j].setNext(fullBucketsArr[j + 1]);
            fullBucketsArr[j + 1].setPrev(fullBucketsArr[j]);
        }
        
        fullBucketsArr[0].setPrev(fullBucketsArr[fullBucketsArr.length - 1]);
        fullBucketsArr[fullBucketsArr.length - 1].setNext(fullBucketsArr[0]);
        
        this.first = fullBucketsArr[0]; // updating first
        this.updateMin(); // updating min - complexity: O(log n). now the heap is a binomial heap.
    }


    /**
     * private HeapNode linkTrees(HeapNode x, HeapNode y)
     * 
     * links x (and its subtree) as a new child of y. (WE DO NOT UPDATE MIN IN THIS FUNCTION)
     * @param x
     * @param y
     * @pre x,y != null.
     * 
     * Complexity: O(1).
     */
    private HeapNode linkTrees(HeapNode x, HeapNode y){ 
        HeapNode smaller = x;
        HeapNode larger = y;

        if (y.getKey() < x.getKey()){ // so that we always link the smaller under the larger.
            smaller = y;
            larger = x;
        }
        
        if (this.first == larger){
            this.first = larger.getNext();
        }
        
        larger.getNext().setPrev(larger.getPrev()); // handling relation of old brothers of larger
        larger.getPrev().setNext(larger.getNext());
        larger.setNext(larger);
        larger.setPrev(larger);
        
        HeapNode oldChild = smaller.getChild(); 
        smaller.setChild(larger); // handling x-y relation
        larger.setParent(smaller);
        if (oldChild != null){ // handling larger-old children relation
            larger.setNext(oldChild);
            larger.setPrev(oldChild.getPrev());
            oldChild.getPrev().setNext(larger);
            oldChild.setPrev(larger);
        }
        
        smaller.setRank(smaller.getRank() + 1); // updating the rank of the new root.
        
        FibonacciHeap.totalLinkes++;
        this.numOfTrees--;

        return smaller;
    }


    /**
     * private void insertNodePointer(HeapNode node)
     * 
     * Inserts node to this heap with original parameter for the Heap Node.
     * @param node
     * 
     * Complexity: O(1).
     */
    private void insertNodePointer(HeapNode node){ 
    	HeapNode newNode = new HeapNode(node, node.getKey()); // new node with pointer to original node in the original heap.
        
        if (!this.isEmpty()){ // updating pointers for previous first and last.
            HeapNode tmpPointerFirst = this.first;
            HeapNode tmpPointerLast = this.first.getPrev();
            
            newNode.setNext(tmpPointerFirst);
            newNode.setPrev(tmpPointerLast);
            tmpPointerFirst.setPrev(newNode);
            tmpPointerLast.setNext(newNode);        
        }
        
        this.first = newNode; // update first
        
        if (this.min == null){ // if there was not a min - update min
            this.min = newNode;
        }
        else{
            if (node.getKey() < this.min.getKey()){ // if there is a min - update min if needed
                this.min = newNode;
            }
        }
        
        this.size++; // update size
        this.numOfTrees++; // update numOfTrees
    }


    /**
     * private void cutChildrenDeleteMin(HeapNode currMin)
     * 
     * @param currMin
     * 
     * Complexity: O(log(n)).
     */
    private void cutChildrenDeleteMin(HeapNode currMin){
        HeapNode minNext = currMin.getNext();
        HeapNode currChild = currMin.getChild();

        
        if (currChild != null){ // if there are any children
            HeapNode nextChild = currChild.getNext();
            boolean flag = true;

            while(flag){ // for each child:
                if (currChild.getNext() == currMin.getChild()){
                    flag = false;
                }
                currChild.setParent(null); // set parent to null
                if (currChild.isMarked()){ // un-mark
                    currChild.setMark(false);
                    this.numOfMarked--;
                }
                this.numOfTrees++; // increase the number of trees by 1. (the child will be a root of a new tree).
                currChild = nextChild;
                nextChild = nextChild.getNext();
            }

            HeapNode firstChild = currMin.getChild();
            HeapNode lastChild = currChild.getPrev();

            firstChild.setPrev(currMin); // handle relations of first and last children so that the children are located to the right of min.
            currMin.setNext(firstChild);
            lastChild.setNext(minNext);
            minNext.setPrev(lastChild);
            
            currMin.setChild(null); // set child of min to null.
        }
    }

    // End of Utility Functions
    
   /**
    * public class HeapNode
    * 
    * If you wish to implement classes other than FibonacciHeap
    * (for example HeapNode), do it in this file, not in another file. 
    *  
    */
    public static class HeapNode{

    	public int key;
    	private int rank;
    	private boolean mark;
    	private HeapNode child;
    	private HeapNode next;
    	private HeapNode prev;
    	private HeapNode parent;
        private HeapNode origin; // I added this for the k min elements function.
    	
    	public HeapNode(int key) {
    		this.key = key;
    		this.rank = 0;
    		this.mark = false;
    		this.child = null;
    		this.next = this;
    		this.prev = this;
    		this.parent = null;
    	}

        public HeapNode(HeapNode node, int key) { // I added this for the k min elements function.
    		this.key = key;
    		this.rank = 0;
    		this.mark = false;
    		this.child = null;
    		this.next = this;
    		this.prev = this;
    		this.parent = null;
            this.origin = node;
    	}

    	public int getKey() {
    		return this.key;
    	}
    	
    	public int getRank() {
    		return this.rank;
    	}
    	
    	public boolean isMarked() {
    		return this.mark;
    	}
    	
    	public HeapNode getChild() {
    		return this.child;
    	}
    	
    	public HeapNode getNext() {
    		return this.next;
    	}
    	
    	public HeapNode getPrev() {
    		return this.prev;
    	}
    	
    	public HeapNode getParent() {
    		return this.parent;
    	}
    	
    	public void setKey(int k) {
    		this.key = k;
    	}
    	
    	public void setRank(int r) {
    		this.rank = r;
    	}
    	
    	public void setMark(boolean m) {
    		this.mark = m;
    	}
    	
    	public void setChild(HeapNode node) {
    		this.child = node;
    	}
    	
    	public void setNext(HeapNode node) {
    		this.next = node;
    	}
    	
    	public void setPrev(HeapNode node) {
    		this.prev = node;
    	}
    	
    	public void setParent(HeapNode node) {
    		this.parent = node;
    	}
    	
        // Utility functions

        private HeapNode getOrigin(){
            return this.origin;
        }

        // end of utility functions
    }
}
