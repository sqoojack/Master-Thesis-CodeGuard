 contract Farm {
  mapping (address => address) public crops;
  event CreateCrop(address indexed owner, address indexed crop);

  function create(address _referrer) external payable {
    require(crops[msg.sender] == address(0));
    crops[msg.sender] = new ProxyCrop(msg.sender, _referrer)._;
    emit CreateCrop(msg.sender, crops[msg.sender]);
  }
}

contract ProxyCrop {
    address public owner;
    bool public disabled;

    constructor(address _owner, address _referrer) public payable {
      owner = _owner;
    }
}