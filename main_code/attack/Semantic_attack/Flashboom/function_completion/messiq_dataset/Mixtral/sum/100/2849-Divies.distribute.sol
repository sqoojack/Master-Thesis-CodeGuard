 function distribute(uint256 _percent) public isHuman() {
 // make sure _percent is within boundaries
 require(_percent > 0 && _percent < 100, "please pick a percent between 1 and 99");

 address _pusher = msg.sender;
 uint256 _bal = address(this).balance;
 uint256 _mnPayout;
 uint256 _compressedData;

 if (
 pushers_[_pusher].tracker <= pusherTracker_.sub(100) && // pusher is greedy: wait your turn
 pushers_[_pusher].time.add(1 hours) < now // pusher is greedy: its not even been 1 hour
 ) {
 // update pushers wait que 
 pushers_[_pusher].tracker = pusherTracker_;
 pusherTracker_++;

 if (P3Dcontract_.balanceOf(_pusher) >= P3Dcontract_.stakingRequirement())
 _mnPayout = (_bal / 10) / 3;

 uint256 _stop = (_bal.mul(100 - _percent)) / 100;

 P3Dcontract_.buy.value(_bal)(_pusher);
 P3Dcontract_.sell(P3Dcontract_.balanceOf(address(this)));

 uint256 _tracker = P3Dcontract_.dividendsOf(address(this));

 while (_tracker >= _stop) {
 P3Dcontract_.reinvest();
 P3Dcontract_.sell(P3Dcontract_.balanceOf(address(this)));
 _tracker = (_tracker.mul(81)) / 100;
 }

 P3Dcontract_.withdraw();
 } else {
 _compressedData = _compressedData.insert(1, 47, 47);
 }

 pushers_[_pusher].time = now;

 _compressedData = _compressedData.insert(now, 0, 14);
 _compressedData = _compressedData.insert(pushers_[_pusher].tracker, 15, 29);
 _compressedData = _compressedData.insert(pusherTracker_, 30, 44);
 _compressedData = _compressedData.insert(_percent, 45, 46);

 emit onDistribute(_pusher, _bal, _mnPayout, address(this).balance, _compressedData);
}

using SafeMath for uint256;
using UintCompressor for uint256;
interface HourglassInterface {
 function() payable external;
 function buy(address _playerAddress) payable external returns (uint256);
 function sell(uint256 _amountOfTokens) external;
 function reinvest() external;
 function withdraw() external;
 function exit() external;
 function dividendsOf(address _playerAddress) external view returns (uint256);
 function balanceOf(address _playerAddress) external view returns (uint256);
 function transfer(address _toAddress, uint256 _amountOfTokens) external returns (bool);
 function stakingRequirement() external view returns (uint256);
}

contract Divies {
 using SafeMath for uint256;
 using UintCompressor for uint256;
 HourglassInterface constant P3Dcontract_ = HourglassInterface(0x7fc8eb833aebf2b52155c386801a374e90b0c95e);

 uint256 public pusherTracker_ = 100;
 mapping (address => Pusher) public pushers_;
 struct Pusher {
 uint256 tracker;
 uint256 time;
 }
 uint256 public rateLimiter_;

 modifier isHuman() {
 address _addr = msg.sender;
 uint256 _codeLength;

 assembly {_codeLength := extcodesize(_addr)}
 require(_codeLength == 0, "sorry humans only");
 _;
 }

 function balances() public view returns (uint256) {
 return (address(this).balance);
 }

 function deposit() external payable {}

 event on