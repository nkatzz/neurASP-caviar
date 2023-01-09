
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

	orientation(mws1gt_3,154,166). coords(mws1gt_3,167,154,166).  visible(mws1gt_3,166). happensAt(walking(mws1gt_3),166). orientation(mws1gt_5,145,166). coords(mws1gt_5,64,87,166).  visible(mws1gt_5,166). happensAt(walking(mws1gt_5),166).
orientation(mws1gt_3,154,167). coords(mws1gt_3,166,154,167).  visible(mws1gt_3,167). happensAt(walking(mws1gt_3),167). orientation(mws1gt_5,145,167). coords(mws1gt_5,64,88,167).  visible(mws1gt_5,167). happensAt(walking(mws1gt_5),167).
orientation(mws1gt_3,154,168). coords(mws1gt_3,164,154,168).  visible(mws1gt_3,168). happensAt(walking(mws1gt_3),168). orientation(mws1gt_5,145,168). coords(mws1gt_5,64,89,168).  visible(mws1gt_5,168). happensAt(walking(mws1gt_5),168).
orientation(mws1gt_3,154,169). coords(mws1gt_3,163,155,169).  visible(mws1gt_3,169). happensAt(walking(mws1gt_3),169). orientation(mws1gt_5,145,169). coords(mws1gt_5,64,90,169).  visible(mws1gt_5,169). happensAt(walking(mws1gt_5),169).
orientation(mws1gt_3,154,170). coords(mws1gt_3,162,155,170).  visible(mws1gt_3,170). happensAt(walking(mws1gt_3),170). orientation(mws1gt_5,145,170). coords(mws1gt_5,64,90,170).  visible(mws1gt_5,170). happensAt(walking(mws1gt_5),170).
orientation(mws1gt_3,154,171). coords(mws1gt_3,160,154,171).  visible(mws1gt_3,171). happensAt(walking(mws1gt_3),171). orientation(mws1gt_5,145,171). coords(mws1gt_5,63,90,171).  visible(mws1gt_5,171). happensAt(walking(mws1gt_5),171).
orientation(mws1gt_3,154,172). coords(mws1gt_3,159,154,172).  visible(mws1gt_3,172). happensAt(walking(mws1gt_3),172). orientation(mws1gt_5,145,172). coords(mws1gt_5,63,90,172).  visible(mws1gt_5,172). happensAt(walking(mws1gt_5),172).
orientation(mws1gt_3,154,173). coords(mws1gt_3,159,154,173).  visible(mws1gt_3,173). happensAt(walking(mws1gt_3),173). orientation(mws1gt_5,145,173). coords(mws1gt_5,62,90,173).  visible(mws1gt_5,173). happensAt(walking(mws1gt_5),173).
orientation(mws1gt_3,154,174). coords(mws1gt_3,158,154,174).  visible(mws1gt_3,174). happensAt(walking(mws1gt_3),174). orientation(mws1gt_5,145,174). coords(mws1gt_5,62,90,174).  visible(mws1gt_5,174). happensAt(walking(mws1gt_5),174).
orientation(mws1gt_3,154,175). coords(mws1gt_3,158,154,175).  visible(mws1gt_3,175). happensAt(walking(mws1gt_3),175). orientation(mws1gt_5,145,175). coords(mws1gt_5,61,90,175).  visible(mws1gt_5,175). happensAt(walking(mws1gt_5),175).
orientation(mws1gt_3,154,176). coords(mws1gt_3,157,154,176).  visible(mws1gt_3,176). happensAt(walking(mws1gt_3),176). orientation(mws1gt_5,145,176). coords(mws1gt_5,61,91,176).  visible(mws1gt_5,176). happensAt(walking(mws1gt_5),176).
orientation(mws1gt_3,154,177). coords(mws1gt_3,157,154,177).  visible(mws1gt_3,177). happensAt(walking(mws1gt_3),177). orientation(mws1gt_5,145,177). coords(mws1gt_5,60,91,177).  visible(mws1gt_5,177). happensAt(walking(mws1gt_5),177).
orientation(mws1gt_3,154,178). coords(mws1gt_3,156,154,178).  visible(mws1gt_3,178). happensAt(walking(mws1gt_3),178). orientation(mws1gt_5,145,178). coords(mws1gt_5,59,93,178).  visible(mws1gt_5,178). happensAt(walking(mws1gt_5),178).
orientation(mws1gt_3,154,179). coords(mws1gt_3,156,154,179).  visible(mws1gt_3,179). happensAt(walking(mws1gt_3),179). orientation(mws1gt_5,145,179). coords(mws1gt_5,59,94,179).  visible(mws1gt_5,179). happensAt(walking(mws1gt_5),179).
orientation(mws1gt_3,154,180). coords(mws1gt_3,156,153,180).  visible(mws1gt_3,180). happensAt(walking(mws1gt_3),180). orientation(mws1gt_5,145,180). coords(mws1gt_5,58,95,180).  visible(mws1gt_5,180). happensAt(walking(mws1gt_5),180).
orientation(mws1gt_3,154,181). coords(mws1gt_3,156,153,181).  visible(mws1gt_3,181). happensAt(walking(mws1gt_3),181). orientation(mws1gt_5,145,181). coords(mws1gt_5,58,96,181).  visible(mws1gt_5,181). happensAt(walking(mws1gt_5),181).
orientation(mws1gt_3,154,182). coords(mws1gt_3,156,153,182).  visible(mws1gt_3,182). happensAt(walking(mws1gt_3),182). orientation(mws1gt_5,145,182). coords(mws1gt_5,58,96,182).  visible(mws1gt_5,182). happensAt(walking(mws1gt_5),182).
orientation(mws1gt_3,154,183). coords(mws1gt_3,155,153,183).  visible(mws1gt_3,183). happensAt(walking(mws1gt_3),183). orientation(mws1gt_5,145,183). coords(mws1gt_5,58,97,183).  visible(mws1gt_5,183). happensAt(walking(mws1gt_5),183).
orientation(mws1gt_3,154,184). coords(mws1gt_3,155,153,184).  visible(mws1gt_3,184). happensAt(walking(mws1gt_3),184). orientation(mws1gt_5,145,184). coords(mws1gt_5,58,97,184).  visible(mws1gt_5,184). happensAt(walking(mws1gt_5),184).
orientation(mws1gt_3,154,185). coords(mws1gt_3,154,153,185).  visible(mws1gt_3,185). happensAt(walking(mws1gt_3),185). orientation(mws1gt_5,145,185). coords(mws1gt_5,57,97,185).  visible(mws1gt_5,185). happensAt(walking(mws1gt_5),185).
orientation(mws1gt_3,154,186). coords(mws1gt_3,154,153,186).  visible(mws1gt_3,186). happensAt(walking(mws1gt_3),186). orientation(mws1gt_5,145,186). coords(mws1gt_5,57,97,186).  visible(mws1gt_5,186). happensAt(walking(mws1gt_5),186).
orientation(mws1gt_3,154,187). coords(mws1gt_3,154,153,187).  visible(mws1gt_3,187). happensAt(walking(mws1gt_3),187). orientation(mws1gt_5,145,187). coords(mws1gt_5,57,98,187).  visible(mws1gt_5,187). happensAt(walking(mws1gt_5),187).
orientation(mws1gt_3,154,188). coords(mws1gt_3,153,153,188).  visible(mws1gt_3,188). happensAt(walking(mws1gt_3),188). orientation(mws1gt_5,145,188). coords(mws1gt_5,56,98,188).  visible(mws1gt_5,188). happensAt(walking(mws1gt_5),188).
orientation(mws1gt_3,154,189). coords(mws1gt_3,153,154,189).  visible(mws1gt_3,189). happensAt(walking(mws1gt_3),189). orientation(mws1gt_5,145,189). coords(mws1gt_5,55,99,189).  visible(mws1gt_5,189). happensAt(walking(mws1gt_5),189).
orientation(mws1gt_3,154,190). coords(mws1gt_3,152,154,190).  visible(mws1gt_3,190). happensAt(walking(mws1gt_3),190). orientation(mws1gt_5,145,190). coords(mws1gt_5,55,99,190).  visible(mws1gt_5,190). happensAt(walking(mws1gt_5),190).
#show holdsAt/2.