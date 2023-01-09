
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

	orientation(fcgt_2,166,174). coords(fcgt_2,33,204,174).  visible(fcgt_2,174). happensAt(walking(fcgt_2),174). orientation(fcgt_3,159,174). coords(fcgt_3,103,163,174).  visible(fcgt_3,174). happensAt(inactive(fcgt_3),174).
orientation(fcgt_2,166,175). coords(fcgt_2,34,204,175).  visible(fcgt_2,175). happensAt(walking(fcgt_2),175). orientation(fcgt_3,159,175). coords(fcgt_3,103,163,175).  visible(fcgt_3,175). happensAt(inactive(fcgt_3),175).
orientation(fcgt_2,166,176). coords(fcgt_2,35,204,176).  visible(fcgt_2,176). happensAt(walking(fcgt_2),176). orientation(fcgt_3,159,176). coords(fcgt_3,103,163,176).  visible(fcgt_3,176). happensAt(inactive(fcgt_3),176).
orientation(fcgt_2,166,177). coords(fcgt_2,35,203,177).  visible(fcgt_2,177). happensAt(walking(fcgt_2),177). orientation(fcgt_3,159,177). coords(fcgt_3,103,163,177).  visible(fcgt_3,177). happensAt(inactive(fcgt_3),177).
orientation(fcgt_2,166,178). coords(fcgt_2,35,203,178).  visible(fcgt_2,178). happensAt(walking(fcgt_2),178). orientation(fcgt_3,159,178). coords(fcgt_3,103,163,178).  visible(fcgt_3,178). happensAt(inactive(fcgt_3),178).
orientation(fcgt_2,166,179). coords(fcgt_2,35,202,179).  visible(fcgt_2,179). happensAt(walking(fcgt_2),179). orientation(fcgt_3,159,179). coords(fcgt_3,103,163,179).  visible(fcgt_3,179). happensAt(inactive(fcgt_3),179).
orientation(fcgt_2,166,180). coords(fcgt_2,36,202,180).  visible(fcgt_2,180). happensAt(walking(fcgt_2),180). orientation(fcgt_3,159,180). coords(fcgt_3,103,163,180).  visible(fcgt_3,180). happensAt(inactive(fcgt_3),180).
orientation(fcgt_2,166,181). coords(fcgt_2,37,201,181).  visible(fcgt_2,181). happensAt(walking(fcgt_2),181). orientation(fcgt_3,159,181). coords(fcgt_3,103,163,181).  visible(fcgt_3,181). happensAt(inactive(fcgt_3),181).
orientation(fcgt_2,166,182). coords(fcgt_2,37,201,182).  visible(fcgt_2,182). happensAt(walking(fcgt_2),182). orientation(fcgt_3,159,182). coords(fcgt_3,103,163,182).  visible(fcgt_3,182). happensAt(inactive(fcgt_3),182).
orientation(fcgt_2,166,183). coords(fcgt_2,37,200,183).  visible(fcgt_2,183). happensAt(walking(fcgt_2),183). orientation(fcgt_3,159,183). coords(fcgt_3,103,163,183).  visible(fcgt_3,183). happensAt(inactive(fcgt_3),183).
orientation(fcgt_2,166,184). coords(fcgt_2,37,201,184).  visible(fcgt_2,184). happensAt(walking(fcgt_2),184). orientation(fcgt_3,159,184). coords(fcgt_3,103,163,184).  visible(fcgt_3,184). happensAt(inactive(fcgt_3),184).
orientation(fcgt_2,166,185). coords(fcgt_2,38,200,185).  visible(fcgt_2,185). happensAt(walking(fcgt_2),185). orientation(fcgt_3,159,185). coords(fcgt_3,103,163,185).  visible(fcgt_3,185). happensAt(inactive(fcgt_3),185).
orientation(fcgt_2,166,186). coords(fcgt_2,38,200,186).  visible(fcgt_2,186). happensAt(walking(fcgt_2),186). orientation(fcgt_3,159,186). coords(fcgt_3,103,163,186).  visible(fcgt_3,186). happensAt(inactive(fcgt_3),186).
orientation(fcgt_2,166,187). coords(fcgt_2,39,199,187).  visible(fcgt_2,187). happensAt(walking(fcgt_2),187). orientation(fcgt_3,159,187). coords(fcgt_3,103,163,187).  visible(fcgt_3,187). happensAt(inactive(fcgt_3),187).
orientation(fcgt_2,166,188). coords(fcgt_2,39,199,188).  visible(fcgt_2,188). happensAt(walking(fcgt_2),188). orientation(fcgt_3,159,188). coords(fcgt_3,103,163,188).  visible(fcgt_3,188). happensAt(inactive(fcgt_3),188).
orientation(fcgt_2,166,189). coords(fcgt_2,40,198,189).  visible(fcgt_2,189). happensAt(walking(fcgt_2),189). orientation(fcgt_3,159,189). coords(fcgt_3,103,163,189).  visible(fcgt_3,189). happensAt(inactive(fcgt_3),189).
orientation(fcgt_2,166,190). coords(fcgt_2,39,198,190).  visible(fcgt_2,190). happensAt(walking(fcgt_2),190). orientation(fcgt_3,159,190). coords(fcgt_3,103,163,190).  visible(fcgt_3,190). happensAt(inactive(fcgt_3),190).
orientation(fcgt_2,166,191). coords(fcgt_2,39,197,191).  visible(fcgt_2,191). happensAt(walking(fcgt_2),191). orientation(fcgt_3,159,191). coords(fcgt_3,103,163,191).  visible(fcgt_3,191). happensAt(inactive(fcgt_3),191).
orientation(fcgt_2,166,192). coords(fcgt_2,39,197,192).  visible(fcgt_2,192). happensAt(walking(fcgt_2),192). orientation(fcgt_3,159,192). coords(fcgt_3,103,163,192).  visible(fcgt_3,192). happensAt(inactive(fcgt_3),192).
orientation(fcgt_2,166,193). coords(fcgt_2,39,197,193).  visible(fcgt_2,193). happensAt(walking(fcgt_2),193). orientation(fcgt_3,159,193). coords(fcgt_3,103,163,193).  visible(fcgt_3,193). happensAt(inactive(fcgt_3),193).
orientation(fcgt_2,166,194). coords(fcgt_2,39,197,194).  visible(fcgt_2,194). happensAt(walking(fcgt_2),194). orientation(fcgt_3,159,194). coords(fcgt_3,103,163,194).  visible(fcgt_3,194). happensAt(inactive(fcgt_3),194).
orientation(fcgt_2,166,195). coords(fcgt_2,39,196,195).  visible(fcgt_2,195). happensAt(walking(fcgt_2),195). orientation(fcgt_3,159,195). coords(fcgt_3,103,163,195).  visible(fcgt_3,195). happensAt(inactive(fcgt_3),195).
orientation(fcgt_2,166,196). coords(fcgt_2,39,195,196).  visible(fcgt_2,196). happensAt(walking(fcgt_2),196). orientation(fcgt_3,159,196). coords(fcgt_3,103,163,196).  visible(fcgt_3,196). happensAt(inactive(fcgt_3),196).
orientation(fcgt_2,166,197). coords(fcgt_2,39,194,197).  visible(fcgt_2,197). happensAt(walking(fcgt_2),197). orientation(fcgt_3,159,197). coords(fcgt_3,103,163,197).  visible(fcgt_3,197). happensAt(inactive(fcgt_3),197).
orientation(fcgt_2,166,198). coords(fcgt_2,40,193,198).  visible(fcgt_2,198). happensAt(walking(fcgt_2),198). orientation(fcgt_3,159,198). coords(fcgt_3,103,163,198).  visible(fcgt_3,198). happensAt(inactive(fcgt_3),198).
#show holdsAt/2.