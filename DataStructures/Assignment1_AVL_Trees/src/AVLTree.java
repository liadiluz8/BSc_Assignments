
/**
 *
 * 
 * Submitting Students:
 * 
 * Student 1:
 * Name: Guy Levy
 * ID: 318645694
 * Moodle username: gl2
 * 
 * Student 2:
 * Name: Liad Iluz
 * ID: 208427435
 * Moodle username: liadiluz
 *
 *  
 */



/**
 *
 * AVLTree
 *
 * An implementation of a� AVL Tree with distinct integer keys and info.
 *
 */

public class AVLTree {

	private IAVLNode root;
	private IAVLNode minimum;
	private IAVLNode maximum;
	private int size;

	public AVLTree() {
		this.root = null;
		this.minimum = null;
		this.maximum = null;
		this.size = 0;
	}

	/**
	 * public boolean empty()
	 *
	 * Returns true if and only if the tree is empty.
	 * Complexity: O(1)
	 */
	public boolean empty() {	
		return (this.size == 0); // checking if tree is empty
	}

	/**
	 * public String search(int k)
	 *
	 * Returns the info of an item with key k if it exists in the tree. otherwise,
	 * returns null.
	 * 
	 * Complexity WC: O(log n)
	 */
	public String search(int k) {
		IAVLNode node = treePosition(k, this.root);	// Complexity WC: O(log n)

		if (node == null) {
			return null;
		}

		if (node.isRealNode() && node.getKey() == k) {
			return node.getValue();
		}

		return null;
	}

	/**
	 * public int insert(int k, String i)
	 *
	 * Inserts an item with key k and info i to the AVL tree. The tree must remain
	 * valid, i.e. keep its invariants. Returns the number of re-balancing
	 * operations, or 0 if no re-balancing operations were necessary. A
	 * promotion/rotation counts as one re-balance operation, double-rotation is
	 * counted as 2. Returns -1 if an item with key k already exists in the tree.
	 * 
	 * Complexity WC: O(log n)
	 */
	public int insert(int k, String s) {
		IAVLNode position = treePosition(k, this.root);	// Complexity WC: O(log n)

		if (position == null) { // The tree was empty.
			IAVLNode newNode = new AVLNode(k, s);
			this.root = newNode;
			this.minimum = newNode;
			this.maximum = newNode;
			this.size = 1;
			return 0;
		}

		if (position.getKey() == k) { // Item already exists.
			return -1;
		}

		IAVLNode newNode = new AVLNode(k, s);

		if (position.getKey() < k) {
			position.setRight(newNode);
		} else {
			position.setLeft(newNode);
		}

		newNode.setParent(position);
		IAVLNode par = position;

		int steps = balanceAfterInsertion(par);		// Complexity WC: O(log n)

		updateSizeNodesFromNode(newNode);			// Complexity WC: O(log n)
		this.updateMin();	// Complexity: O(log n)
		this.updateMax();	// Complexity: O(log n)
		this.size += 1;

		return steps;
	}

	/**
	 * public int delete(int k)
	 *
	 * Deletes an item with key k from the binary tree, if it is there. The tree
	 * must remain valid, i.e. keep its invariants. Returns the number of
	 * re-balancing operations, or 0 if no re-balancing operations were necessary. A
	 * promotion/rotation counts as one re-balance operation, double-rotation is
	 * counted as 2. Returns -1 if an item with key k was not found in the tree.
	 * 
	 * Complexity: O(log n)
	 */
	public int delete(int k) {
		IAVLNode node = treePosition(k, this.root);		// Complexity WC: O(log n)

		if (node == null) { 	// if the key k is not in the tree
			return -1;
		}

		if (node.getKey() != k) {
			return -1;
		}

		String kindOfNode = getKindOfNode(node);	// Complexity: O(1)
		String typeNode = kindOfNode.split("_")[0]; // leaf, unary, binary. Complexity: O(1)
		String dirNode = kindOfNode.split("_")[1]; 	// root, left(son), right(son). Complexity: O(1)

		IAVLNode parentOfDeleted = deleteNode(node, typeNode, dirNode);	// Complexity WC: O(log n)

		int balancedSteps = balanceAfterDeletion(parentOfDeleted);	// Complexity WC: O(log n)

		if (parentOfDeleted != null) {
			updateSizeNodesFromNode(parentOfDeleted.getLeft());		// Complexity WC: O(log n)
			updateSizeNodesFromNode(parentOfDeleted.getRight());	// Complexity WC: O(log n)
		}
		
		this.updateMin(); 	// Complexity: O(log n)
		this.updateMax();	// Complexity: O(log n)
		this.size -= 1;

		return balancedSteps;

	}

	/**
	 * public String min()
	 *
	 * Returns the info of the item with the smallest key in the tree, or null if
	 * the tree is empty.
	 * 
	 * Complexity: O(1)
	 */
	public String min() {
		if (!this.empty()) {
			return this.minimum.getValue();
		}
		return null;
	}

	/**
	 * public String max()
	 *
	 * Returns the info of the item with the largest key in the tree, or null if the
	 * tree is empty.
	 * 
	 * Complexity: O(1)
	 */
	public String max() {
		if (!this.empty()) {
			return this.maximum.getValue();
		}
		return null;
	}

	/**
	 * public int[] keysToArray()
	 *
	 * Returns a sorted array which contains all keys in the tree, or an empty array
	 * if the tree is empty.
	 * 
	 * Complexity: O(n)
	 */
	public int[] keysToArray() {

		int[] keysArr = new int[this.size];

		if (this.empty()) {
			return keysArr;
		}

		int i = 0; // Index of last keysArr item.
		IAVLNode node = this.minimum;

		while (node != null) {			// Number of iterations: O(n)
			keysArr[i] = node.getKey();
			node = getSuccessor(node);	// Complexity: WC: O(log n), Amortized: O(1)
			i++;
		}

		return keysArr;
	}

	/**
	 * public String[] infoToArray()
	 *
	 * Returns an array which contains all info in the tree, sorted by their
	 * respective keys, or an empty array if the tree is empty.
	 * 
	 * Complexity: O(n)
	 */
	public String[] infoToArray() {
		String[] infoArr = new String[this.size];

		if (this.empty()) {
			return infoArr;
		}

		int i = 0; // Index of last infoArr item.
		IAVLNode node = this.minimum;

		while (node != null) {				// Number of iterations: O(n)
			infoArr[i] = node.getValue();
			node = getSuccessor(node);		// Complexity: WC: O(log n), Amortized: O(1)
			i++;
		}

		return infoArr;
	}

	/**
	 * public int size()
	 *
	 * Returns the number of nodes in the tree.
	 * 
	 * Complexity: O(1)
	 */
	public int size() {
		return this.size;
	}

	/**
	 * public int getRoot()
	 *
	 * Returns the root AVL node, or null if the tree is empty
	 * 
	 * Complexity: O(1)
	 */
	public IAVLNode getRoot() {
		if (!this.empty()) {
			return this.root;
		}
		return null;
	}

	/**
	 * public AVLTree[] split(int x)
	 *
	 * splits the tree into 2 trees according to the key x. Returns an array [t1,
	 * t2] with two AVL trees. keys(t1) < x < keys(t2).
	 * 
	 * precondition: search(x) != null (i.e. you can also assume that the tree is
	 * not empty) postcondition: none
	 * 
	 * Complexity WC: O(log n)
	 */
	public AVLTree[] split(int x) {
		IAVLNode xNode = treePosition(x, this.root);	// Complexity WC: O(log n)
		
		AVLTree[] splitTrees = new AVLTree[2];
		AVLTree smallTree, bigTree;
		AVLTree otherSmallT, otherBigT;

		smallTree = new AVLTree();
		bigTree = new AVLTree();
		otherSmallT = new AVLTree();
		otherBigT = new AVLTree();

		if (xNode.getLeft().isRealNode()) {
			smallTree.root = xNode.getLeft();
			xNode.getLeft().setParent(null);
			smallTree.size = xNode.getLeft().getSizeNode();
		} else {
			smallTree = new AVLTree();
		}

		if (xNode.getRight().isRealNode()) {
			bigTree.root = xNode.getRight();
			xNode.getRight().setParent(null);
			bigTree.size = xNode.getRight().getSizeNode();
		} else {
			bigTree = new AVLTree();
		}

		IAVLNode par = xNode.getParent();
		IAVLNode temp = xNode;
		IAVLNode nextPar;

		while (par != null) {				// Number of iterations: O(log n) = O(this.height)
			nextPar = par.getParent();

			if (par.getLeft() == temp) {
				if (par.getRight().isRealNode()) {
					otherBigT.root = par.getRight();
					par.getRight().setParent(null);
					otherBigT.size = par.getRight().getSizeNode();
				} else {
					otherBigT = new AVLTree();
				}

				// par is free node - it's a free leaf
				par.setParent(null);
				par.setRight(new AVLNode(par));
				par.setLeft(new AVLNode(par));
				par.setSizeNode(1);
				par.setHeight(0);

				bigTree.join(par, otherBigT);		// Complexity: O(|bigTree.root.height - otherBigTree.root.height|)
				
			} else if (par.getRight() == temp) {
				if (par.getLeft().isRealNode()) {
					otherSmallT.root = par.getLeft();
					par.getLeft().setParent(null);
					otherSmallT.size = par.getLeft().getSizeNode();
				} else {
					otherSmallT = new AVLTree();
				}

				// par is free node - it's a free leaf
				par.setParent(null);
				par.setRight(new AVLNode(par));
				par.setLeft(new AVLNode(par));
				par.setSizeNode(1);
				par.setHeight(0);

				smallTree.join(par, otherSmallT);	// Complexity: O(|smallTree.root.height - otherSmallT.root.height|)
			}

			temp = par;
			par = nextPar;
		}

		// UPDATE MIN and MAX of each TREE
		smallTree.updateMin();	// Complexity WC: O(log n)
		smallTree.updateMax();	// Complexity WC: O(log n)
		bigTree.updateMin();	// Complexity WC: O(log n)
		bigTree.updateMax();	// Complexity WC: O(log n)

		splitTrees[0] = smallTree;
		splitTrees[1] = bigTree;

		return splitTrees;
	}

	/**
	 * public int join(IAVLNode x, AVLTree t)
	 *
	 * joins t and x with the tree. Returns the complexity of the operation
	 * (|tree.rank - t.rank| + 1).
	 *
	 * precondition: keys(t) < x < keys() or keys(t) > x > keys(). t/tree might be
	 * empty (rank = -1). postcondition: none
	 * 
	 * Complexity WC: O(|this.root.height - t.root.height|)
	 */
	public int join(IAVLNode x, AVLTree t) {
		int complex = 0;

		if (t.empty() && this.empty()) {
			this.root = x;
			this.size = 1;
			this.minimum = x;
			this.maximum = x;

			complex = 1;
		}

		else if (t.empty() && !this.empty()) {
			if (x.getKey() < this.root.getKey()) {
				IAVLNode min = findMinNode(this.root); // Complexity: O(this.height) = O(|this.root.height - t.root.height|)
				min.setLeft(x);
				x.setParent(min);
				this.minimum = x; 					// update minimum of this
				updateSizeNodesFromNode(min); 			// Complexity: O(this.height) = O(|this.root.height - t.root.height|)
				balanceAfterInsertion(min);				// Complexity: O(this.height) = O(|this.root.height - t.root.height|)
			} else {
				IAVLNode max = findMaxNode(this.root);	// Complexity: O(this.height) = O(|this.root.height - t.root.height|)
				max.setRight(x);
				x.setParent(max);
				this.maximum = x; 					// update maximum of this
				updateSizeNodesFromNode(max); 			// Complexity: O(this.height) = O(|this.root.height - t.root.height|)
				balanceAfterInsertion(max);				// Complexity: O(this.height) = O(|this.root.height - t.root.height|)
			}
			
			if (this.getRoot().getParent() != null) {	// the root of this was changed after balance
				updateRoot(this);						// Complexity WC: O(this.height) = O(|this.root.height - t.root.height|)
			}
				
			this.size = this.getRoot().getSizeNode();

			complex = this.root.getHeight();
		}

		else if (this.empty() && !t.empty()) {
			if (x.getKey() < t.root.getKey()) {
				IAVLNode min = findMinNode(t.root); 	// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
				min.setLeft(x);
				x.setParent(min);
				t.minimum = x; 						// update minimum of t
				updateSizeNodesFromNode(min); 			// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
				balanceAfterInsertion(min);				// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
				
			} else {
				IAVLNode max = findMaxNode(t.root); 	// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
				max.setRight(x);
				x.setParent(max);
				t.maximum = x; 						// update maximum of t
				updateSizeNodesFromNode(max); 			// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
				balanceAfterInsertion(max);				// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
			}
						
			if (t.getRoot().getParent() != null) {	// the root of t was changed after balance
				updateRoot(t);							// Complexity: O(t.height) = O(|this.root.height - t.root.height|)
			}
			
			t.size = t.getRoot().getSizeNode();
			
			this.root = t.getRoot();
			this.minimum = t.getMinimum();
			this.maximum = t.getMaximum();
			this.size = t.getSize();

			complex = this.root.getHeight();
		}

		else if (!this.empty() && !t.empty()) {
			complex = Math.abs(this.root.getHeight() - t.getRoot().getHeight()) + 1;

			AVLTree bigTree = this;
			AVLTree smallTree = t;

			if (smallTree.getRoot().getHeight() > bigTree.getRoot().getHeight()) {
				bigTree = t;
				smallTree = this;
			}

			IAVLNode curr = bigTree.getRoot();

			if (bigTree.getRoot().getKey() < smallTree.getRoot().getKey()) {
				while (curr.getHeight() > smallTree.getRoot().getHeight()) {	
												// Number of iterations = Complexity:  O(|this.root.height - t.root.height|)
					curr = curr.getRight();
				}

				x.setHeight(smallTree.getRoot().getHeight() + 1); 	// setting x's new height

				x.setRight(smallTree.getRoot()); 		// smallroot is left son of x
				smallTree.getRoot().setParent(x);

				x.setParent(curr.getParent()); 			// curr.parent is parent of x
				if (curr.getParent() != null) { 		// For case curr is the root of bigTree
					curr.getParent().setRight(x);
				} else { 								// case of smallTree's height equals to bigTree's height
					bigTree.root = x;
				}

				x.setLeft(curr); 		// curr is right son of x
				curr.setParent(x);

				this.minimum = bigTree.getMinimum();
				this.maximum = smallTree.getMaximum();

			} else {
				while (curr.getHeight() > smallTree.getRoot().getHeight()) {
												// Number of iterations = Complexity:  O(|this.root.height - t.root.height|)
					curr = curr.getLeft();
				}

				x.setHeight(smallTree.getRoot().getHeight() + 1); // setting x's new height

				x.setLeft(smallTree.getRoot()); 		// smallTree.root is left son of x
				smallTree.getRoot().setParent(x);

				x.setParent(curr.getParent()); 			// curr.parent is parent of x
				if (curr.getParent() != null) { 		// For case curr is the root of bigTree
					curr.getParent().setLeft(x);
				} else { 								// case of smallTree's height equals to bigTree's height
					bigTree.root = x;
				}

				x.setRight(curr); 		// curr is right son of x
				curr.setParent(x);

				this.minimum = smallTree.getMinimum();
				this.maximum = bigTree.getMaximum();
			}

			this.root = bigTree.getRoot();
			
			updateSizeNodesFromNode(x);				// Complexity:  O(|this.root.height - t.root.height|)
			balanceAfterInsertion(x.getParent());	// Complexity WC:  O(|this.root.height - t.root.height|)
			updateSizeNodesFromNode(x);				// Complexity:  O(|this.root.height - t.root.height|)

			if (this.getRoot().getParent() != null) {	
				updateRoot(this);					// Complexity WC:  O(|this.root.height - t.root.height|)
			}
			
			this.size = this.root.getSizeNode();
		}

		return complex;
	}

	// Our utility functions:

	/**
	 * The function updateRoot updates the root of the tree while tree.root.getParent() != null
	 * @param tree - AVL Tree
	 * Complexity WC:  O(log n)
	 */
	private static void updateRoot(AVLTree tree) {
		IAVLNode curr = tree.getRoot();
		IAVLNode newRoot = curr;
		
		while (curr.getParent() != null) {		// Number of iterations = Complexity WC:  O(log n)
			newRoot = curr.getParent();
			curr = newRoot;		
		}
		
		tree.root = newRoot;
	}

	public int getSize() {
		return this.size;
	}

	public IAVLNode getMaximum() {
		return this.maximum;
	}

	public IAVLNode getMinimum() {
		return this.minimum;
	}

	/**
	 * 
	 * @param key - integer
	 * @param curr - IAVLNode node
	 * @return null if curr = null, IAVLNode node: (node.getKey() == k) if key is in the tree, 
	 * and node that supposes to be the parent of k if isn't.
	 * 
	 * Complexity WC:  O(log n)
	 */
	public IAVLNode treePosition(int key, IAVLNode curr) {
		if (curr == null) {
			return null;
		}

		IAVLNode temp = curr;

		while (curr.isRealNode()) {		// Number of iterations = Complexity WC:  O(log n)
			temp = curr;
			if (curr.getKey() == key) {
				return curr;
			} else {
				if (curr.getKey() < key) {
					curr = curr.getRight();
				} else {
					curr = curr.getLeft();
				}
			}
		}

		return temp;
	}

	/**
	 * The function getSuccessor returns the successor of node if it's real node
	 * pre: node != null
	 * @param node - IAVLNode node we want to get the successor of it
	 * @return IAVLNode s = the successor of node or null if node.isRealNode=false
	 * 
	 * Complexity WC:  O(log n)
	 */
	private IAVLNode getSuccessor(IAVLNode node) {
		if (!node.isRealNode()) {
			return null;
		}

		IAVLNode curr = node;
		IAVLNode tempParent = curr.getParent();

		if (curr.getRight().isRealNode()) {
			return findMinNode(curr.getRight());	// Complexity WC:  O(log n)
		}

		if (tempParent != null) {
			while ((tempParent != null) && (curr == tempParent.getRight())) {	// Complexity WC:  O(log n)
				curr = tempParent;
				tempParent = curr.getParent();
			}
		}

		return tempParent;
	}

	/**
	 * The function getPredecessor returns the predecessor of node if it's real node
	 * pre: node != null
	 * @param node - IAVLNode node we want to get the predecessor of it
	 * @return IAVLNode s = the predecessor of node or null if node.isRealNode=false
	 * 
	 * Complexity WC:  O(log n)
	 */
	private IAVLNode getPredecessor(IAVLNode node) {
		if (!node.isRealNode()) {
			return null;
		}

		IAVLNode curr = node;
		IAVLNode tempParent = curr.getParent();

		if (curr.getLeft().isRealNode()) {
			return findMaxNode(curr.getLeft());		// Complexity WC:  O(log n)
		}

		if (tempParent != null) {
			while ((tempParent != null) && (curr == tempParent.getLeft())) {	// Complexity WC:  O(log n)
				curr = tempParent;
				tempParent = curr.getParent();
			}
		}

		return tempParent;
	}

	/**
	 * The function deletes node from its tree. 
	 * @param node - The node of the tree we want to delete.
	 * @param typeNode - The type of node: leaf, unary or binary.
	 * @param dirNode - The direction of the node with reference to its parent
	 * pre: the node is given is real AVLNode in the tree.
	 * @return the parent of the node we deleted.
	 * 
	 * Complexity WC: O(log n)
	 */
	private IAVLNode deleteNode(IAVLNode node, String typeNode, String dirNode) {
		IAVLNode par = node.getParent();

		// node is a leaf
		if (typeNode.equals("Leaf")) {
			if (dirNode.equals("root")) { // deleting the root and initializing empty tree
				this.root = null;

				// par is null
			} else {
				IAVLNode terminal = new AVLNode(par);
				if (dirNode.equals("right")) {
					par.setRight(terminal);
				} else if (dirNode.equals("left")) {
					par.setLeft(terminal);
				}
			}
		}

		// node is an unary node
		else if (typeNode.equals("Unary")) {
			if (dirNode.equals("root")) {
				if (node.getLeft().isRealNode()) {
					this.root = node.getLeft();
					node.getLeft().setParent(par);
				}
				if (node.getRight().isRealNode()) {
					this.root = node.getRight();
					node.getRight().setParent(par);
				}

				// par is null
			} else {
				IAVLNode nodeSon; // the only son of the unary node
				if (node.getLeft().isRealNode()) {
					nodeSon = node.getLeft();
				} else {
					nodeSon = node.getRight();
				}

				if (dirNode.equals("right")) {
					par.setRight(nodeSon);
				} else if (dirNode.equals("left")) {
					par.setLeft(nodeSon);
				}

				nodeSon.setParent(par);
			}
		}

		// node is a binary node
		else {
			IAVLNode y = getSuccessor(node); 	// y has no left child. Complexity WC: O(log n)
			IAVLNode tempYpar = y.getParent();	// this is the node we wish to return
			IAVLNode ySon = y.getRight(); 		// ySon is real node or is not real node
			IAVLNode yPar = y.getParent();

			// removing y from the tree
			ySon.setParent(yPar);
			if (yPar.getLeft() == y) {
				yPar.setLeft(ySon);
			} else {
				yPar.setRight(ySon);
			}

			// replacing y and node
			y.setParent(par);
			if (dirNode.equals("root")) {
				this.root = y;
				// par is null
			} else if (dirNode.equals("right")) {
				par.setRight(y);
			} else if (dirNode.equals("left")) {
				par.setLeft(y);
			}

			y.setLeft(node.getLeft());
			node.getLeft().setParent(y);

			if (node.getRight() != y) { // avoiding of self pointing as a son - situation: y (the successor of node) is
										// right son of node
				y.setRight(node.getRight());
				node.getRight().setParent(y);
			}

			// maintain y properties: height and sizeNode by node's properties.
			y.setHeight(node.getHeight());
			y.setSizeNode(node.getSizeNode());
			
			par = tempYpar;
		}
		
		return par;
	}

	/**
	 * The function updateSizeNodesFromNode updates the sizes of all nodes from given node
	 * 	all the way up until up node.getParent=null
	 * Complexity: O(root.height - node.height + 1). WC (if node is a leaf):  O(log n). 
	 */
	private static void updateSizeNodesFromNode(IAVLNode node) {
		while (node != null) {		// Number of iterations = Complexity WC:  O(log n)
			updateSizeNode(node);	// Complexity:  O(1)
			node = node.getParent();
		}
	}

	/**
	 * T�he function updateSizeNode updates the sizeNode of node
	 * @param node
	 * post: if node.isRealNode=false >> node.sizeNode=0, 
	 *       else >> node.sizeNode = node.getLeft().sizeNode + node.getRight().sizeNode + 1
	 * Complexity: O(1).
	 */
	private static void updateSizeNode(IAVLNode node) {
		if (node == null) {
			return;
		}
		if (!node.isRealNode()) {
			node.setSizeNode(0);
			return;
		}

		// this Node is real Node
		node.setSizeNode(node.getLeft().getSizeNode() + node.getRight().getSizeNode() + 1);
	}

	/**
	 * The function updateMax updates the maximum of the tree this by finding the maximum
	 * Complexity WC:  O(log n)
	 */
	private void updateMax() {
		IAVLNode curr = this.root;
		this.maximum = findMaxNode(curr);	// Complexity WC:  O(log n)
	}

	/**
	 * The function updateMin updates the minimum of the tree this by finding the minimum
	 * Complexity WC:  O(log n)
	 */
	private void updateMin() {
		IAVLNode curr = this.root;
		this.minimum = findMinNode(curr);	// Complexity WC:  O(log n)
	}

	/**
	 * The function findMinNode finds and returns the node with the minimum key from given IAVLNode node,
	 *  and returns null if node=null.
	 * Complexity WC:  O(log n)
	 */
	public IAVLNode findMinNode(IAVLNode node) {	
		IAVLNode curr = node;

		if (curr == null) {
			return curr;
		}

		while (curr.getLeft().isRealNode()) {	// Complexity WC:  O(log n)
			curr = curr.getLeft();
		}

		return curr;
	}

	/**
	 * The function findMaxNode finds and returns the node with the maximum key from given IAVLNode node,
	 * and returns null if node=null.
	 * Complexity WC:  O(log n)
	 */
	public IAVLNode findMaxNode(IAVLNode node) {
		IAVLNode curr = node;

		if (curr == null) {
			return curr;
		}

		while (curr.getRight().isRealNode()) {	// Complexity WC:  O(log n)
			curr = curr.getRight();
		}

		return curr;
	}

	/**
	 * The function getKindOfNode classifies the type of real node:
	 * leaf, unary or binary; and root, right son or left son of its parent.
	 * @param node - IAVLNode node
	 * @return the classify of node and returns one of the followings: Leaf_root / Leaf_right / Leaf_left /
	 * Unary_root / Unary_right / Unary_left / Binary_root / Binary_right / Binary_left.
	 * returns null if node=null.
	 * Complexity: O(1)
	 */
	private String getKindOfNode(IAVLNode node) { 
		String kindNode = "";
		
		if (node == null) {
			return kindNode;
		}
		
		if (isLeaf(node)) {		// Complexity: O(1)
			if (node.getParent() == null) // if the leaf is the root
			{
				kindNode = "Leaf_root";
				return kindNode;
			}
			if (node.getParent().getRight().getKey() == node.getKey()) { // if it is right son leaf
				kindNode = "Leaf_right";
				return kindNode;
			}
			if (node.getParent().getLeft().getKey() == node.getKey()) { // if it is left son leaf
				kindNode = "Leaf_left";
				return kindNode;
			}
		}

		if (isUnary(node)) {	// Complexity: O(1)
			if (node.getParent() == null) // if the unary node is the root
			{
				kindNode = "Unary_root";
				return kindNode;
			}
			if (node.getParent().getRight().getKey() == node.getKey()) { // if it is right son unary node
				kindNode = "Unary_right";
				return kindNode;
			}
			if (node.getParent().getLeft().getKey() == node.getKey()) { // if it is left son unary node
				kindNode = "Unary_left";
				return kindNode;
			}
		}

		// this is binary node
		if (node.getParent() == null) { // if the binary node is the root
			kindNode = "Binary_root";
			return kindNode;
		}
		if (node.getParent().getRight().getKey() == node.getKey()) { // if it is right son binary node
			kindNode = "Binary_right";
			return kindNode;
		}
		if (node.getParent().getLeft().getKey() == node.getKey()) { // if it is left son binary node
			kindNode = "Binary_left";
			return kindNode;
		}

		return null;
	}

	/**
	 * @param node - IAVLNode node
	 * @return true iff node is unary node (has only one real son)
	 * 
	 * Complexity: O(1).
	 */
	private boolean isUnary(IAVLNode node) {
		if ((node.getLeft().isRealNode() && !node.getRight().isRealNode())
				|| (!node.getLeft().isRealNode() && node.getRight().isRealNode()))
			return true;

		return false;
	}

	/**
	 * @param node - IAVLNode node
	 * @return true iff node is a leaf in the tree (has no real sons)
	 * 
	 * Complexity: O(1).
	 */
	private boolean isLeaf(IAVLNode node) { 
		if (!node.getLeft().isRealNode() && !node.getRight().isRealNode())
			return true;

		return false;
	}

	/**
	 * Rotates right currSone.
	 * @param currSon
	 * 
	 * Complexity: O(1).
	 */
	private void rotateRight(IAVLNode currSon) {
		IAVLNode x = currSon;
		IAVLNode y = x.getParent();
		IAVLNode b = x.getRight();

		x.setParent(y.getParent()); // Setting y's parent and x relation.
		if (y.getParent() != null) {
			if (y == y.getParent().getLeft()) {
				y.getParent().setLeft(x);
			} else {
				y.getParent().setRight(x);
			}
		} else { // y is the root of the tree, so now x is the root
			this.root = x;
		}

		x.setRight(y);
		y.setParent(x);

		y.setLeft(b);
		b.setParent(y);
		
		updateSizeNode(y); // Maintain y.sizeNode after rotate. Complexity: O(1).
		updateSizeNode(x); // Maintain x.sizeNode after rotate. Complexity: O(1).
	}

	/**
	 * Rotates left currSone.
	 * @param currSon
	 * 
	 * Complexity: O(1).
	 */
	private void rotateLeft(IAVLNode currSon) {
		IAVLNode y = currSon;
		IAVLNode x = y.getParent();
		IAVLNode b = y.getLeft();

		y.setParent(x.getParent()); // Setting x's parent and y relation.
		if (x.getParent() != null) {
			if (x == x.getParent().getLeft()) {
				x.getParent().setLeft(y);
			} else {
				x.getParent().setRight(y);
			}
		} else { // x is the root of the tree, so now y is the root
			this.root = y;
		}

		y.setLeft(x);
		x.setParent(y);

		x.setRight(b);
		b.setParent(x);

		updateSizeNode(x); // Maintain x.sizeNode after rotate. Complexity: O(1).
		updateSizeNode(y); // Maintain y.sizeNode after rotate. Complexity: O(1).
	}

	/**
	 * Demote height of node.
	 * @param node
	 * pre: node.isRealNode()==true
	 * post: node.height = (pre)node.height - 1
	 * 
	 * Complexity: O(1).
	 */
	private void demote(IAVLNode node) {
		node.setHeight(node.getHeight() - 1);
	}

	/**
	 * Promote height of node.
	 * @param node
	 * pre: node.isRealNode()==true
	 * post: node.height = (pre)node.height + 1
	 * 
	 * Complexity: O(1).
	 */
	private void promote(IAVLNode node) {
		node.setHeight(node.getHeight() + 1);
	}

	/**
	 * The function balanceAfterInsertion balances the tree after insert from given node newParent for keeping the tree as AVL tree.
	 * @param newParent
	 * @return number of balance operations: #promote + #demote + #rotate left + #rotateRight
	 * 
	 * Complexity WC: O(newParent.height) = O(log n).
	 */
	private int balanceAfterInsertion(IAVLNode newParent) {
		int opNum = 0;
		IAVLNode z = newParent;

		if (z == null) { // in case x is the root
			return 0;
		}

		while (!isLegalAVLNode(z)) {	// Complexity isLegalAVLNode = O(1). Number of iterations: O(parentOfDeleted.height) = O(log n).
			if (z.getHeight() - z.getLeft().getHeight() == 0) { // If 0 edge is from left
				IAVLNode x = z.getLeft();
				IAVLNode y = z.getRight();
				IAVLNode a = x.getLeft();
				IAVLNode b = x.getRight();

				if (z.getHeight() - y.getHeight() == 1) { // Case 1
					promote(z);
					opNum += 1;
				}

				else if ((x.getHeight() - a.getHeight() == 1) && (x.getHeight() - b.getHeight() == 2)) { // Case 2
					rotateRight(x);
					demote(z);
					opNum += 2; // Balance complete
				}

				else if ((x.getHeight() - a.getHeight() == 2) && (x.getHeight() - b.getHeight() == 1)) { // Case 3
					rotateLeft(b);
					rotateRight(b);
					demote(x);
					demote(z);
					promote(b);
					opNum += 5; // Balance complete
				}
				
				else if ((x.getHeight() - a.getHeight() == 1) && (x.getHeight() - b.getHeight() == 1)) { // Case join
					rotateRight(x);
					promote(x);
					opNum += 2; 
					z = z.getParent();
				}
			}

			else if (z.getHeight() - z.getRight().getHeight() == 0) { // If 0 edge is from right
				IAVLNode x = z.getRight();
				IAVLNode y = z.getLeft();
				IAVLNode a = x.getRight();
				IAVLNode b = x.getLeft();

				if (z.getHeight() - y.getHeight() == 1) { // Case 1
					promote(z);
					opNum += 1;
				}

				else if ((x.getHeight() - a.getHeight() == 1) && (x.getHeight() - b.getHeight() == 2)) { // Case 2
					rotateLeft(x);
					demote(z);
					opNum += 2; // Balance complete
				}

				else if ((x.getHeight() - a.getHeight() == 2) && (x.getHeight() - b.getHeight() == 1)) { // Case 3
					rotateRight(b);
					rotateLeft(b);
					demote(x);
					demote(z);
					promote(b);
					opNum += 5; // Balance complete
				}
				else if ((x.getHeight() - a.getHeight() == 1) && (x.getHeight() - b.getHeight() == 1)) { // Case join
					rotateLeft(x);
					promote(x);
					opNum += 2; 
					z = z.getParent();
				}
			}

			z = z.getParent();
			if (z == null) {
				break;
			}

		}

		return opNum;
	}

	/**
	 * The function balanceAfterDeletion balances the tree after delete from given node parentOfDeleted for keeping the tree as AVL tree.
	 * @param newParent
	 * @return number of balance operations: #promote + #demote + #rotate left + #rotateRight
	 * 
	 * Complexity WC: O(parentOfDeleted.height) = O(log n).
	 */
	private int balanceAfterDeletion(IAVLNode parentOfDeleted) {
		int opNum = 0;
		IAVLNode z = parentOfDeleted;

		if (z == null) { // in case x is the root
			return 0;
		}

		while (!isLegalAVLNode(z)) {	// Complexity isLegalAVLNode = O(1). Number of iterations: O(parentOfDeleted.height) = O(log n).
			if ((z.getHeight() - z.getLeft().getHeight() == 2) && (z.getHeight() - z.getRight().getHeight() == 2)) { // Case
																														// 1
				demote(z);
				opNum += 1;
			}

			else if (z.getHeight() - z.getLeft().getHeight() == 3) { // If 3 edge is in left son
				IAVLNode x = z.getLeft();
				IAVLNode y = z.getRight();
				IAVLNode a = y.getLeft();
				IAVLNode b = y.getRight();

				if ((y.getHeight() - a.getHeight() == 1) && (y.getHeight() - b.getHeight() == 1)) { // Case 2 - 3 left
					rotateLeft(y);
					demote(z);
					promote(y);
					opNum += 3; // Balance complete
				}

				else if ((y.getHeight() - a.getHeight() == 2) && (y.getHeight() - b.getHeight() == 1)) { // Case 3 - 3
																											// left
					rotateLeft(y);
					demote(z);
					demote(z);
					opNum += 3;
					z = z.getParent();
				}

				else if ((y.getHeight() - a.getHeight() == 1) && (y.getHeight() - b.getHeight() == 2)) { // Case 4 - 3
																											// left
					rotateRight(a);
					rotateLeft(a);
					demote(z);
					demote(z);
					promote(a);
					demote(y);
					opNum += 6;
					z = z.getParent();
				}
			}

			else if (z.getHeight() - z.getRight().getHeight() == 3) { // If y is right son
				IAVLNode x = z.getRight();
				IAVLNode y = z.getLeft();
				IAVLNode a = y.getRight();
				IAVLNode b = y.getLeft();

				if ((y.getHeight() - a.getHeight() == 1) && (y.getHeight() - b.getHeight() == 1)) { // Case 2 - 3 right
					rotateRight(y);
					demote(z);
					promote(y);
					opNum += 3; // Balance complete
				}

				else if ((y.getHeight() - a.getHeight() == 2) && (y.getHeight() - b.getHeight() == 1)) { // Case 3 - 3
																											// right
					rotateRight(y);
					demote(z);
					demote(z);
					opNum += 3;
					z = z.getParent();
				}

				else if ((y.getHeight() - a.getHeight() == 1) && (y.getHeight() - b.getHeight() == 2)) { // Case 4 - 3
																											// right
					rotateLeft(a);
					rotateRight(a);
					demote(z);
					demote(z);
					promote(a);
					demote(y);
					opNum += 6;
					z = z.getParent();
				}
			}

			z = z.getParent();
			if (z == null) {
				break;
			}
		}

		return opNum;
	}

	/**
	 * The function isLegalAVLNode checks if node is legaal AVL node.
	 * @param node
	 * pre: node.isRealNode() == true.
	 * @return true iff node is legal AVL node (ranks types: 1-1 / 2-1 / 1-2).
	 * 
	 * Complexity: O(1).
	 */
	private boolean isLegalAVLNode(IAVLNode node) { 
		int currHeight = node.getHeight();
		int leftHeight = node.getLeft().getHeight();
		int rightHeiht = node.getRight().getHeight();

		if (currHeight - leftHeight == 2 && currHeight - rightHeiht == 1) {
			return true;
		}

		if (currHeight - leftHeight == 1 && currHeight - rightHeiht == 2) {
			return true;
		}

		if (currHeight - leftHeight == 1 && currHeight - rightHeiht == 1) {
			return true;
		}

		return false;
	}

	// End of utility functions
	
	
	
	
	// functions for analysis
	/*

	public int fingerTreeInsert(int k, String s) {
		if (this.empty()) { // The tree was empty.
			IAVLNode newNode = new AVLNode(k, s);
			this.root = newNode;
			this.minimum = newNode;
			this.maximum = newNode;
			this.size = 1;
			return 1;
		}
		
		int opNum = 1;
		IAVLNode maxNode = this.maximum;
		IAVLNode curr = maxNode;
		
		while (curr.getParent() != null && curr.getParent().getKey() > k) {
			opNum++;
			curr = curr.getParent();
		}
		
		IAVLNode temp = curr;

		while (curr.isRealNode()) {		// Number of iterations = Complexity WC:  O(log n)
			temp = curr;
			//if (curr.getKey() == k) {
			//	break;
			//} else {
				if (curr.getKey() < k) {
					curr = curr.getRight();
				} else {
					curr = curr.getLeft();
				}
			//}
			opNum++;
		}
		

		IAVLNode position = temp;
		
		
		// if (position.getKey() == k) { // Item already exists.
		// 	return -1;
		// }

		IAVLNode newNode = new AVLNode(k, s);

		if (position.getKey() < k) {
			position.setRight(newNode);
		} else {
			position.setLeft(newNode);
		}

		newNode.setParent(position);
		IAVLNode par = position;

		int steps = balanceAfterInsertion(par);

		updateSizeNodesFromNode(newNode);
		this.updateMin();
		this.updateMax();
		this.size += 1;

		return opNum;
	}
	
		
	public AVLTree[] splitForAnalasis(int x) {
		IAVLNode xNode = treePosition(x, this.root);
		AVLTree[] splitTrees = new AVLTree[2];
		AVLTree smallTree, bigTree;
		AVLTree otherSmallT, otherBigT;

		smallTree = new AVLTree();
		bigTree = new AVLTree();
		otherSmallT = new AVLTree();
		otherBigT = new AVLTree();

		if (xNode.getLeft().isRealNode()) {
			smallTree.root = xNode.getLeft();
			xNode.getLeft().setParent(null);
			smallTree.size = xNode.getLeft().getSizeNode();
		} else {
			smallTree = new AVLTree();
		}

		if (xNode.getRight().isRealNode()) {
			bigTree.root = xNode.getRight();
			xNode.getRight().setParent(null);
			bigTree.size = xNode.getRight().getSizeNode();
		} else {
			bigTree = new AVLTree();
		}

		IAVLNode par = xNode.getParent();
		IAVLNode temp = xNode;
		IAVLNode nextPar;
		
		int costJoin;
		
		while (par != null) {
			nextPar = par.getParent();

			if (par.getLeft() == temp) {
				if (par.getRight().isRealNode()) {
					otherBigT.root = par.getRight();
					par.getRight().setParent(null);
					otherBigT.size = par.getRight().getSizeNode();
				} else {
					otherBigT = new AVLTree();
				}
				// temp.setParent(null); ????????

				// par is free node - it's a free leaf
				par.setParent(null);
				par.setRight(new AVLNode(par));
				par.setLeft(new AVLNode(par));
				par.setSizeNode(1);
				par.setHeight(0);

				costJoin = bigTree.join(par, otherBigT);
				updateCosts(costJoin);
				
			} else if (par.getRight() == temp) {
				if (par.getLeft().isRealNode()) {
					otherSmallT.root = par.getLeft();
					par.getLeft().setParent(null);
					otherSmallT.size = par.getLeft().getSizeNode();
				} else {
					otherSmallT = new AVLTree();
				}
				// temp.setParent(null); ????????

				// par is free node - it's a free leaf
				par.setParent(null);
				par.setRight(new AVLNode(par));
				par.setLeft(new AVLNode(par));
				par.setSizeNode(1);
				par.setHeight(0);

				costJoin = smallTree.join(par, otherSmallT);
				updateCosts(costJoin);
			}

			temp = par;
			par = nextPar;
		}

		// UPDATE MIN and MAX of each TREE
		// updateSizeNodesFromNode()
		smallTree.updateMin();
		smallTree.updateMax();
		bigTree.updateMin();
		bigTree.updateMax();

		splitTrees[0] = smallTree;
		splitTrees[1] = bigTree;

		return splitTrees;
	}

	private static void updateCosts(int costJoin) {
		TesterToDelete.numOfJoin++;
		TesterToDelete.sumOfJoin += costJoin;
		
		if (TesterToDelete.maxJoin < costJoin) {
			TesterToDelete.maxJoin = costJoin;
		}
	}
	
	*/

	// End of functions for analysis

	



	/**
	 * public interface IAVLNode ! Do not delete or modify this - otherwise all
	 * tests will fail !
	 */
	public interface IAVLNode {
		public int getKey(); // Returns node's key (for virtual node return -1).

		public String getValue(); // Returns node's value [info], for virtual node returns null.

		public void setLeft(IAVLNode node); // Sets left child.

		public IAVLNode getLeft(); // Returns left child, if there is no left child returns null.

		public void setRight(IAVLNode node); // Sets right child.

		public IAVLNode getRight(); // Returns right child, if there is no right child return null.

		public void setParent(IAVLNode node); // Sets parent.

		public IAVLNode getParent(); // Returns the parent, if there is no parent return null.

		public boolean isRealNode(); // Returns True if this is a non-virtual AVL node.

		public void setHeight(int height); // Sets the height of the node.

		public int getHeight(); // Returns the height of the node (-1 for virtual nodes).

		public int getSizeNode(); // Returns the number of nodes in the sub-tree of node (0 for virtual nodes).

		public void setSizeNode(int s); // Sets sizeNode to s.
	}

	/**
	 * public class AVLNode
	 *
	 * If you wish to implement classes other than AVLTree (for example AVLNode), do
	 * it in this file, not in another file.
	 * 
	 * This class can and MUST be modified (It must implement IAVLNode).
	 */
	public class AVLNode implements IAVLNode {

		private String info;
		private int key;
		private IAVLNode rightSon;
		private IAVLNode leftSon;
		private IAVLNode parent;
		private int height;
		private boolean isReal;
		private int sizeNode; // The number of nodes in the sub-tree from this node

		public AVLNode(int key, String info) {
			this.info = info;
			this.key = key;
			this.rightSon = new AVLNode(this); // For every node we initiate non real sons
			this.leftSon = new AVLNode(this);
			this.parent = null;
			this.height = 0;
			this.isReal = true;
			this.sizeNode = 1;
		}

		public AVLNode(IAVLNode parent) {
			this.key = -1;
			this.parent = parent;
			this.height = -1;
			this.isReal = false;
			this.sizeNode = 0;
		}

		public int getSizeNode() {
			return this.sizeNode;
		}

		public void setSizeNode(int s) {
			this.sizeNode = s;
		}

		public int getKey() {
			if (this.isReal) {
				return this.key;
			}
			return -1; // if node is virtual, returning -1

		}

		public String getValue() {
			if (this.isReal) {
				return this.info;
			}
			return null; // if node is virtual, returning null
		}

		public void setLeft(IAVLNode node) {
			this.leftSon = node;
		}

		public IAVLNode getLeft() {
			return this.leftSon;
		}

		public void setRight(IAVLNode node) {
			this.rightSon = node;
		}

		public IAVLNode getRight() {
			return this.rightSon;
		}

		public void setParent(IAVLNode node) {
			this.parent = node;
		}

		public IAVLNode getParent() {
			return this.parent;
		}

		public boolean isRealNode() {
			return this.isReal;
		}

		public void setHeight(int height) {
			this.height = height;
		}

		public int getHeight() {
			if (this.isReal) {
				return this.height;
			}
			return -1;
		}
	}

}
