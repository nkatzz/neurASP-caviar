
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(rsfgt_0,160,175). coords(rsfgt_0,30,161,175).  visible(rsfgt_0,175). happensAt(walking(rsfgt_0),175). orientation(rsfgt_1,150,175). coords(rsfgt_1,203,160,175).  visible(rsfgt_1,175). happensAt(walking(rsfgt_1),175).
orientation(rsfgt_0,160,176). coords(rsfgt_0,30,161,176).  visible(rsfgt_0,176). happensAt(walking(rsfgt_0),176). orientation(rsfgt_1,150,176). coords(rsfgt_1,206,163,176).  visible(rsfgt_1,176). happensAt(walking(rsfgt_1),176).
orientation(rsfgt_0,160,177). coords(rsfgt_0,30,162,177).  visible(rsfgt_0,177). happensAt(walking(rsfgt_0),177). orientation(rsfgt_1,150,177). coords(rsfgt_1,209,167,177).  visible(rsfgt_1,177). happensAt(walking(rsfgt_1),177).
orientation(rsfgt_0,160,178). coords(rsfgt_0,26,164,178).  visible(rsfgt_0,178). happensAt(walking(rsfgt_0),178). orientation(rsfgt_1,150,178). coords(rsfgt_1,211,169,178).  visible(rsfgt_1,178). happensAt(walking(rsfgt_1),178).
orientation(rsfgt_0,160,179). coords(rsfgt_0,26,164,179).  visible(rsfgt_0,179). happensAt(walking(rsfgt_0),179). orientation(rsfgt_1,150,179). coords(rsfgt_1,213,170,179).  visible(rsfgt_1,179). happensAt(walking(rsfgt_1),179).
orientation(rsfgt_0,160,180). coords(rsfgt_0,26,164,180).  visible(rsfgt_0,180). happensAt(walking(rsfgt_0),180). orientation(rsfgt_1,150,180). coords(rsfgt_1,215,172,180).  visible(rsfgt_1,180). happensAt(walking(rsfgt_1),180).
orientation(rsfgt_0,160,181). coords(rsfgt_0,26,164,181).  visible(rsfgt_0,181). happensAt(walking(rsfgt_0),181). orientation(rsfgt_1,150,181). coords(rsfgt_1,215,172,181).  visible(rsfgt_1,181). happensAt(walking(rsfgt_1),181).
orientation(rsfgt_0,160,182). coords(rsfgt_0,26,164,182).  visible(rsfgt_0,182). happensAt(walking(rsfgt_0),182). orientation(rsfgt_1,150,182). coords(rsfgt_1,216,174,182).  visible(rsfgt_1,182). happensAt(walking(rsfgt_1),182).
orientation(rsfgt_0,160,183). coords(rsfgt_0,26,164,183).  visible(rsfgt_0,183). happensAt(walking(rsfgt_0),183). orientation(rsfgt_1,150,183). coords(rsfgt_1,217,174,183).  visible(rsfgt_1,183). happensAt(walking(rsfgt_1),183).
orientation(rsfgt_0,160,184). coords(rsfgt_0,26,164,184).  visible(rsfgt_0,184). happensAt(walking(rsfgt_0),184). orientation(rsfgt_1,150,184). coords(rsfgt_1,218,176,184).  visible(rsfgt_1,184). happensAt(walking(rsfgt_1),184).
orientation(rsfgt_0,160,185). coords(rsfgt_0,26,164,185).  visible(rsfgt_0,185). happensAt(walking(rsfgt_0),185). orientation(rsfgt_1,150,185). coords(rsfgt_1,218,177,185).  visible(rsfgt_1,185). happensAt(walking(rsfgt_1),185).
orientation(rsfgt_0,160,186). coords(rsfgt_0,26,164,186).  visible(rsfgt_0,186). happensAt(walking(rsfgt_0),186). orientation(rsfgt_1,150,186). coords(rsfgt_1,218,177,186).  visible(rsfgt_1,186). happensAt(walking(rsfgt_1),186).
orientation(rsfgt_0,160,187). coords(rsfgt_0,26,164,187).  visible(rsfgt_0,187). happensAt(walking(rsfgt_0),187). orientation(rsfgt_1,150,187). coords(rsfgt_1,221,178,187).  visible(rsfgt_1,187). happensAt(walking(rsfgt_1),187).
orientation(rsfgt_0,160,188). coords(rsfgt_0,26,164,188).  visible(rsfgt_0,188). happensAt(walking(rsfgt_0),188). orientation(rsfgt_1,150,188). coords(rsfgt_1,221,178,188).  visible(rsfgt_1,188). happensAt(walking(rsfgt_1),188).
orientation(rsfgt_0,160,189). coords(rsfgt_0,26,166,189).  visible(rsfgt_0,189). happensAt(walking(rsfgt_0),189). orientation(rsfgt_1,150,189). coords(rsfgt_1,228,182,189).  visible(rsfgt_1,189). happensAt(walking(rsfgt_1),189).
orientation(rsfgt_0,160,190). coords(rsfgt_0,26,166,190).  visible(rsfgt_0,190). happensAt(walking(rsfgt_0),190). orientation(rsfgt_1,150,190). coords(rsfgt_1,233,185,190).  visible(rsfgt_1,190). happensAt(walking(rsfgt_1),190).
orientation(rsfgt_0,160,191). coords(rsfgt_0,26,166,191).  visible(rsfgt_0,191). happensAt(walking(rsfgt_0),191). orientation(rsfgt_1,150,191). coords(rsfgt_1,235,188,191).  visible(rsfgt_1,191). happensAt(walking(rsfgt_1),191).
orientation(rsfgt_0,160,192). coords(rsfgt_0,26,166,192).  visible(rsfgt_0,192). happensAt(walking(rsfgt_0),192). orientation(rsfgt_1,150,192). coords(rsfgt_1,237,189,192).  visible(rsfgt_1,192). happensAt(walking(rsfgt_1),192).
orientation(rsfgt_0,160,193). coords(rsfgt_0,26,166,193).  visible(rsfgt_0,193). happensAt(walking(rsfgt_0),193). orientation(rsfgt_1,150,193). coords(rsfgt_1,240,191,193).  visible(rsfgt_1,193). happensAt(walking(rsfgt_1),193).
orientation(rsfgt_0,160,194). coords(rsfgt_0,26,166,194).  visible(rsfgt_0,194). happensAt(walking(rsfgt_0),194). orientation(rsfgt_1,150,194). coords(rsfgt_1,240,192,194).  visible(rsfgt_1,194). happensAt(walking(rsfgt_1),194).
orientation(rsfgt_0,160,195). coords(rsfgt_0,26,166,195).  visible(rsfgt_0,195). happensAt(walking(rsfgt_0),195). orientation(rsfgt_1,150,195). coords(rsfgt_1,243,192,195).  visible(rsfgt_1,195). happensAt(walking(rsfgt_1),195).
orientation(rsfgt_0,160,196). coords(rsfgt_0,26,166,196).  visible(rsfgt_0,196). happensAt(walking(rsfgt_0),196). orientation(rsfgt_1,150,196). coords(rsfgt_1,244,192,196).  visible(rsfgt_1,196). happensAt(walking(rsfgt_1),196).
orientation(rsfgt_0,160,197). coords(rsfgt_0,26,166,197).  visible(rsfgt_0,197). happensAt(walking(rsfgt_0),197). orientation(rsfgt_1,150,197). coords(rsfgt_1,244,196,197).  visible(rsfgt_1,197). happensAt(walking(rsfgt_1),197).
orientation(rsfgt_0,160,198). coords(rsfgt_0,26,166,198).  visible(rsfgt_0,198). happensAt(walking(rsfgt_0),198). orientation(rsfgt_1,170,198). coords(rsfgt_1,245,200,198).  visible(rsfgt_1,198). happensAt(walking(rsfgt_1),198).
orientation(rsfgt_0,160,199). coords(rsfgt_0,26,166,199).  happensAt(disappear(rsfgt_0),199). happensAt(walking(rsfgt_0),199). orientation(rsfgt_1,170,199). coords(rsfgt_1,246,202,199).  visible(rsfgt_1,199). happensAt(walking(rsfgt_1),199).
#show holdsAt/2.