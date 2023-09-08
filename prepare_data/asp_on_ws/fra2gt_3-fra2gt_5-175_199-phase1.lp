
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

	orientation(fra2gt_3,161,175). coords(fra2gt_3,37,193,175).  visible(fra2gt_3,175). happensAt(walking(fra2gt_3),175). orientation(fra2gt_5,150,175). coords(fra2gt_5,29,117,175).  visible(fra2gt_5,175). happensAt(walking(fra2gt_5),175).
orientation(fra2gt_3,161,176). coords(fra2gt_3,38,192,176).  visible(fra2gt_3,176). happensAt(walking(fra2gt_3),176). orientation(fra2gt_5,150,176). coords(fra2gt_5,29,117,176).  visible(fra2gt_5,176). happensAt(walking(fra2gt_5),176).
orientation(fra2gt_3,161,177). coords(fra2gt_3,38,192,177).  visible(fra2gt_3,177). happensAt(walking(fra2gt_3),177). orientation(fra2gt_5,150,177). coords(fra2gt_5,29,117,177).  visible(fra2gt_5,177). happensAt(walking(fra2gt_5),177).
orientation(fra2gt_3,161,178). coords(fra2gt_3,38,191,178).  visible(fra2gt_3,178). happensAt(walking(fra2gt_3),178). orientation(fra2gt_5,150,178). coords(fra2gt_5,29,116,178).  visible(fra2gt_5,178). happensAt(walking(fra2gt_5),178).
orientation(fra2gt_3,161,179). coords(fra2gt_3,39,191,179).  visible(fra2gt_3,179). happensAt(walking(fra2gt_3),179). orientation(fra2gt_5,150,179). coords(fra2gt_5,29,116,179).  visible(fra2gt_5,179). happensAt(walking(fra2gt_5),179).
orientation(fra2gt_3,161,180). coords(fra2gt_3,39,190,180).  visible(fra2gt_3,180). happensAt(walking(fra2gt_3),180). orientation(fra2gt_5,150,180). coords(fra2gt_5,29,116,180).  visible(fra2gt_5,180). happensAt(walking(fra2gt_5),180).
orientation(fra2gt_3,161,181). coords(fra2gt_3,39,190,181).  visible(fra2gt_3,181). happensAt(walking(fra2gt_3),181). orientation(fra2gt_5,150,181). coords(fra2gt_5,29,116,181).  visible(fra2gt_5,181). happensAt(walking(fra2gt_5),181).
orientation(fra2gt_3,161,182). coords(fra2gt_3,39,190,182).  visible(fra2gt_3,182). happensAt(walking(fra2gt_3),182). orientation(fra2gt_5,150,182). coords(fra2gt_5,28,115,182).  visible(fra2gt_5,182). happensAt(walking(fra2gt_5),182).
orientation(fra2gt_3,161,183). coords(fra2gt_3,39,189,183).  visible(fra2gt_3,183). happensAt(walking(fra2gt_3),183). orientation(fra2gt_5,150,183). coords(fra2gt_5,28,115,183).  visible(fra2gt_5,183). happensAt(walking(fra2gt_5),183).
orientation(fra2gt_3,161,184). coords(fra2gt_3,39,188,184).  visible(fra2gt_3,184). happensAt(walking(fra2gt_3),184). orientation(fra2gt_5,150,184). coords(fra2gt_5,28,114,184).  visible(fra2gt_5,184). happensAt(walking(fra2gt_5),184).
orientation(fra2gt_3,161,185). coords(fra2gt_3,39,187,185).  visible(fra2gt_3,185). happensAt(walking(fra2gt_3),185). orientation(fra2gt_5,150,185). coords(fra2gt_5,28,114,185).  visible(fra2gt_5,185). happensAt(walking(fra2gt_5),185).
orientation(fra2gt_3,161,186). coords(fra2gt_3,39,187,186).  visible(fra2gt_3,186). happensAt(walking(fra2gt_3),186). orientation(fra2gt_5,150,186). coords(fra2gt_5,28,113,186).  visible(fra2gt_5,186). happensAt(walking(fra2gt_5),186).
orientation(fra2gt_3,161,187). coords(fra2gt_3,39,185,187).  visible(fra2gt_3,187). happensAt(walking(fra2gt_3),187). orientation(fra2gt_5,150,187). coords(fra2gt_5,28,113,187).  visible(fra2gt_5,187). happensAt(walking(fra2gt_5),187).
orientation(fra2gt_3,161,188). coords(fra2gt_3,39,184,188).  visible(fra2gt_3,188). happensAt(walking(fra2gt_3),188). orientation(fra2gt_5,150,188). coords(fra2gt_5,28,113,188).  visible(fra2gt_5,188). happensAt(walking(fra2gt_5),188).
orientation(fra2gt_3,161,189). coords(fra2gt_3,40,183,189).  visible(fra2gt_3,189). happensAt(walking(fra2gt_3),189). orientation(fra2gt_5,150,189). coords(fra2gt_5,30,113,189).  visible(fra2gt_5,189). happensAt(walking(fra2gt_5),189).
orientation(fra2gt_3,161,190). coords(fra2gt_3,40,183,190).  visible(fra2gt_3,190). happensAt(walking(fra2gt_3),190). orientation(fra2gt_5,150,190). coords(fra2gt_5,30,113,190).  visible(fra2gt_5,190). happensAt(walking(fra2gt_5),190).
orientation(fra2gt_3,161,191). coords(fra2gt_3,41,183,191).  visible(fra2gt_3,191). happensAt(walking(fra2gt_3),191). orientation(fra2gt_5,150,191). coords(fra2gt_5,30,113,191).  visible(fra2gt_5,191). happensAt(walking(fra2gt_5),191).
orientation(fra2gt_3,161,192). coords(fra2gt_3,41,183,192).  visible(fra2gt_3,192). happensAt(walking(fra2gt_3),192). orientation(fra2gt_5,150,192). coords(fra2gt_5,30,112,192).  visible(fra2gt_5,192). happensAt(walking(fra2gt_5),192).
orientation(fra2gt_3,161,193). coords(fra2gt_3,42,182,193).  visible(fra2gt_3,193). happensAt(walking(fra2gt_3),193). orientation(fra2gt_5,150,193). coords(fra2gt_5,30,112,193).  visible(fra2gt_5,193). happensAt(walking(fra2gt_5),193).
orientation(fra2gt_3,161,194). coords(fra2gt_3,42,182,194).  visible(fra2gt_3,194). happensAt(walking(fra2gt_3),194). orientation(fra2gt_5,150,194). coords(fra2gt_5,30,112,194).  visible(fra2gt_5,194). happensAt(walking(fra2gt_5),194).
orientation(fra2gt_3,161,195). coords(fra2gt_3,42,182,195).  visible(fra2gt_3,195). happensAt(walking(fra2gt_3),195). orientation(fra2gt_5,150,195). coords(fra2gt_5,30,110,195).  visible(fra2gt_5,195). happensAt(walking(fra2gt_5),195).
orientation(fra2gt_3,161,196). coords(fra2gt_3,42,181,196).  visible(fra2gt_3,196). happensAt(walking(fra2gt_3),196). orientation(fra2gt_5,150,196). coords(fra2gt_5,30,110,196).  visible(fra2gt_5,196). happensAt(walking(fra2gt_5),196).
orientation(fra2gt_3,161,197). coords(fra2gt_3,42,181,197).  visible(fra2gt_3,197). happensAt(walking(fra2gt_3),197). orientation(fra2gt_5,150,197). coords(fra2gt_5,30,108,197).  visible(fra2gt_5,197). happensAt(walking(fra2gt_5),197).
orientation(fra2gt_3,161,198). coords(fra2gt_3,42,179,198).  visible(fra2gt_3,198). happensAt(walking(fra2gt_3),198). orientation(fra2gt_5,150,198). coords(fra2gt_5,30,108,198).  visible(fra2gt_5,198). happensAt(walking(fra2gt_5),198).
orientation(fra2gt_3,161,199). coords(fra2gt_3,43,178,199).  visible(fra2gt_3,199). happensAt(walking(fra2gt_3),199). orientation(fra2gt_5,150,199). coords(fra2gt_5,30,108,199).  visible(fra2gt_5,199). happensAt(walking(fra2gt_5),199).
#show holdsAt/2.