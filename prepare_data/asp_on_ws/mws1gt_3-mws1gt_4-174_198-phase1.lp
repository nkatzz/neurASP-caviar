
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

	orientation(mws1gt_3,154,174). coords(mws1gt_3,158,154,174).  visible(mws1gt_3,174). happensAt(walking(mws1gt_3),174). orientation(mws1gt_4,0,174). coords(mws1gt_4,29,121,174).  visible(mws1gt_4,174). happensAt(walking(mws1gt_4),174).
orientation(mws1gt_3,154,175). coords(mws1gt_3,158,154,175).  visible(mws1gt_3,175). happensAt(walking(mws1gt_3),175). orientation(mws1gt_4,0,175). coords(mws1gt_4,29,120,175).  visible(mws1gt_4,175). happensAt(walking(mws1gt_4),175).
orientation(mws1gt_3,154,176). coords(mws1gt_3,157,154,176).  visible(mws1gt_3,176). happensAt(walking(mws1gt_3),176). orientation(mws1gt_4,0,176). coords(mws1gt_4,29,120,176).  visible(mws1gt_4,176). happensAt(walking(mws1gt_4),176).
orientation(mws1gt_3,154,177). coords(mws1gt_3,157,154,177).  visible(mws1gt_3,177). happensAt(walking(mws1gt_3),177). orientation(mws1gt_4,0,177). coords(mws1gt_4,29,120,177).  visible(mws1gt_4,177). happensAt(walking(mws1gt_4),177).
orientation(mws1gt_3,154,178). coords(mws1gt_3,156,154,178).  visible(mws1gt_3,178). happensAt(walking(mws1gt_3),178). orientation(mws1gt_4,0,178). coords(mws1gt_4,29,119,178).  visible(mws1gt_4,178). happensAt(walking(mws1gt_4),178).
orientation(mws1gt_3,154,179). coords(mws1gt_3,156,154,179).  visible(mws1gt_3,179). happensAt(walking(mws1gt_3),179). orientation(mws1gt_4,0,179). coords(mws1gt_4,29,119,179).  visible(mws1gt_4,179). happensAt(walking(mws1gt_4),179).
orientation(mws1gt_3,154,180). coords(mws1gt_3,156,153,180).  visible(mws1gt_3,180). happensAt(walking(mws1gt_3),180). orientation(mws1gt_4,0,180). coords(mws1gt_4,30,118,180).  visible(mws1gt_4,180). happensAt(walking(mws1gt_4),180).
orientation(mws1gt_3,154,181). coords(mws1gt_3,156,153,181).  visible(mws1gt_3,181). happensAt(walking(mws1gt_3),181). orientation(mws1gt_4,0,181). coords(mws1gt_4,30,118,181).  visible(mws1gt_4,181). happensAt(walking(mws1gt_4),181).
orientation(mws1gt_3,154,182). coords(mws1gt_3,156,153,182).  visible(mws1gt_3,182). happensAt(walking(mws1gt_3),182). orientation(mws1gt_4,0,182). coords(mws1gt_4,30,118,182).  visible(mws1gt_4,182). happensAt(walking(mws1gt_4),182).
orientation(mws1gt_3,154,183). coords(mws1gt_3,155,153,183).  visible(mws1gt_3,183). happensAt(walking(mws1gt_3),183). orientation(mws1gt_4,0,183). coords(mws1gt_4,30,117,183).  visible(mws1gt_4,183). happensAt(walking(mws1gt_4),183).
orientation(mws1gt_3,154,184). coords(mws1gt_3,155,153,184).  visible(mws1gt_3,184). happensAt(walking(mws1gt_3),184). orientation(mws1gt_4,0,184). coords(mws1gt_4,30,117,184).  visible(mws1gt_4,184). happensAt(walking(mws1gt_4),184).
orientation(mws1gt_3,154,185). coords(mws1gt_3,154,153,185).  visible(mws1gt_3,185). happensAt(walking(mws1gt_3),185). orientation(mws1gt_4,0,185). coords(mws1gt_4,30,117,185).  visible(mws1gt_4,185). happensAt(walking(mws1gt_4),185).
orientation(mws1gt_3,154,186). coords(mws1gt_3,154,153,186).  visible(mws1gt_3,186). happensAt(walking(mws1gt_3),186). orientation(mws1gt_4,0,186). coords(mws1gt_4,30,116,186).  visible(mws1gt_4,186). happensAt(walking(mws1gt_4),186).
orientation(mws1gt_3,154,187). coords(mws1gt_3,154,153,187).  visible(mws1gt_3,187). happensAt(walking(mws1gt_3),187). orientation(mws1gt_4,0,187). coords(mws1gt_4,30,115,187).  visible(mws1gt_4,187). happensAt(walking(mws1gt_4),187).
orientation(mws1gt_3,154,188). coords(mws1gt_3,153,153,188).  visible(mws1gt_3,188). happensAt(walking(mws1gt_3),188). orientation(mws1gt_4,0,188). coords(mws1gt_4,30,115,188).  visible(mws1gt_4,188). happensAt(walking(mws1gt_4),188).
orientation(mws1gt_3,154,189). coords(mws1gt_3,153,154,189).  visible(mws1gt_3,189). happensAt(walking(mws1gt_3),189). orientation(mws1gt_4,0,189). coords(mws1gt_4,31,114,189).  visible(mws1gt_4,189). happensAt(walking(mws1gt_4),189).
orientation(mws1gt_3,154,190). coords(mws1gt_3,152,154,190).  visible(mws1gt_3,190). happensAt(walking(mws1gt_3),190). orientation(mws1gt_4,0,190). coords(mws1gt_4,31,114,190).  visible(mws1gt_4,190). happensAt(walking(mws1gt_4),190).
orientation(mws1gt_3,158,191). coords(mws1gt_3,152,154,191).  visible(mws1gt_3,191). happensAt(walking(mws1gt_3),191). orientation(mws1gt_4,0,191). coords(mws1gt_4,31,114,191).  visible(mws1gt_4,191). happensAt(walking(mws1gt_4),191).
orientation(mws1gt_3,160,192). coords(mws1gt_3,151,154,192).  visible(mws1gt_3,192). happensAt(walking(mws1gt_3),192). orientation(mws1gt_4,0,192). coords(mws1gt_4,31,114,192).  visible(mws1gt_4,192). happensAt(walking(mws1gt_4),192).
orientation(mws1gt_3,163,193). coords(mws1gt_3,150,154,193).  visible(mws1gt_3,193). happensAt(walking(mws1gt_3),193). orientation(mws1gt_4,0,193). coords(mws1gt_4,31,113,193).  visible(mws1gt_4,193). happensAt(walking(mws1gt_4),193).
orientation(mws1gt_3,162,194). coords(mws1gt_3,149,154,194).  visible(mws1gt_3,194). happensAt(walking(mws1gt_3),194). orientation(mws1gt_4,0,194). coords(mws1gt_4,31,113,194).  visible(mws1gt_4,194). happensAt(walking(mws1gt_4),194).
orientation(mws1gt_3,161,195). coords(mws1gt_3,147,154,195).  visible(mws1gt_3,195). happensAt(walking(mws1gt_3),195). orientation(mws1gt_4,0,195). coords(mws1gt_4,31,113,195).  visible(mws1gt_4,195). happensAt(walking(mws1gt_4),195).
orientation(mws1gt_3,162,196). coords(mws1gt_3,146,154,196).  visible(mws1gt_3,196). happensAt(walking(mws1gt_3),196). orientation(mws1gt_4,0,196). coords(mws1gt_4,32,113,196).  visible(mws1gt_4,196). happensAt(walking(mws1gt_4),196).
orientation(mws1gt_3,162,197). coords(mws1gt_3,144,154,197).  visible(mws1gt_3,197). happensAt(walking(mws1gt_3),197). orientation(mws1gt_4,0,197). coords(mws1gt_4,32,112,197).  visible(mws1gt_4,197). happensAt(walking(mws1gt_4),197).
orientation(mws1gt_3,162,198). coords(mws1gt_3,143,154,198).  visible(mws1gt_3,198). happensAt(walking(mws1gt_3),198). orientation(mws1gt_4,0,198). coords(mws1gt_4,32,112,198).  visible(mws1gt_4,198). happensAt(walking(mws1gt_4),198).
#show holdsAt/2.