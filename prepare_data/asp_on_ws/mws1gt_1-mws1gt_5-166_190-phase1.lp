
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

	orientation(mws1gt_1,155,166). coords(mws1gt_1,142,143,166).  visible(mws1gt_1,166). happensAt(walking(mws1gt_1),166). orientation(mws1gt_5,145,166). coords(mws1gt_5,64,87,166).  visible(mws1gt_5,166). happensAt(walking(mws1gt_5),166).
orientation(mws1gt_1,155,167). coords(mws1gt_1,142,143,167).  visible(mws1gt_1,167). happensAt(walking(mws1gt_1),167). orientation(mws1gt_5,145,167). coords(mws1gt_5,64,88,167).  visible(mws1gt_5,167). happensAt(walking(mws1gt_5),167).
orientation(mws1gt_1,155,168). coords(mws1gt_1,148,142,168).  visible(mws1gt_1,168). happensAt(walking(mws1gt_1),168). orientation(mws1gt_5,145,168). coords(mws1gt_5,64,89,168).  visible(mws1gt_5,168). happensAt(walking(mws1gt_5),168).
orientation(mws1gt_1,155,169). coords(mws1gt_1,148,142,169).  visible(mws1gt_1,169). happensAt(walking(mws1gt_1),169). orientation(mws1gt_5,145,169). coords(mws1gt_5,64,90,169).  visible(mws1gt_5,169). happensAt(walking(mws1gt_5),169).
orientation(mws1gt_1,155,170). coords(mws1gt_1,148,142,170).  visible(mws1gt_1,170). happensAt(walking(mws1gt_1),170). orientation(mws1gt_5,145,170). coords(mws1gt_5,64,90,170).  visible(mws1gt_5,170). happensAt(walking(mws1gt_5),170).
orientation(mws1gt_1,155,171). coords(mws1gt_1,149,141,171).  visible(mws1gt_1,171). happensAt(walking(mws1gt_1),171). orientation(mws1gt_5,145,171). coords(mws1gt_5,63,90,171).  visible(mws1gt_5,171). happensAt(walking(mws1gt_5),171).
orientation(mws1gt_1,155,172). coords(mws1gt_1,149,140,172).  visible(mws1gt_1,172). happensAt(walking(mws1gt_1),172). orientation(mws1gt_5,145,172). coords(mws1gt_5,63,90,172).  visible(mws1gt_5,172). happensAt(walking(mws1gt_5),172).
orientation(mws1gt_1,155,173). coords(mws1gt_1,149,140,173).  visible(mws1gt_1,173). happensAt(walking(mws1gt_1),173). orientation(mws1gt_5,145,173). coords(mws1gt_5,62,90,173).  visible(mws1gt_5,173). happensAt(walking(mws1gt_5),173).
orientation(mws1gt_1,155,174). coords(mws1gt_1,150,139,174).  visible(mws1gt_1,174). happensAt(walking(mws1gt_1),174). orientation(mws1gt_5,145,174). coords(mws1gt_5,62,90,174).  visible(mws1gt_5,174). happensAt(walking(mws1gt_5),174).
orientation(mws1gt_1,155,175). coords(mws1gt_1,150,139,175).  visible(mws1gt_1,175). happensAt(walking(mws1gt_1),175). orientation(mws1gt_5,145,175). coords(mws1gt_5,61,90,175).  visible(mws1gt_5,175). happensAt(walking(mws1gt_5),175).
orientation(mws1gt_1,155,176). coords(mws1gt_1,150,138,176).  visible(mws1gt_1,176). happensAt(walking(mws1gt_1),176). orientation(mws1gt_5,145,176). coords(mws1gt_5,61,91,176).  visible(mws1gt_5,176). happensAt(walking(mws1gt_5),176).
orientation(mws1gt_1,155,177). coords(mws1gt_1,151,137,177).  visible(mws1gt_1,177). happensAt(walking(mws1gt_1),177). orientation(mws1gt_5,145,177). coords(mws1gt_5,60,91,177).  visible(mws1gt_5,177). happensAt(walking(mws1gt_5),177).
orientation(mws1gt_1,155,178). coords(mws1gt_1,151,136,178).  visible(mws1gt_1,178). happensAt(walking(mws1gt_1),178). orientation(mws1gt_5,145,178). coords(mws1gt_5,59,93,178).  visible(mws1gt_5,178). happensAt(walking(mws1gt_5),178).
orientation(mws1gt_1,155,179). coords(mws1gt_1,151,136,179).  visible(mws1gt_1,179). happensAt(walking(mws1gt_1),179). orientation(mws1gt_5,145,179). coords(mws1gt_5,59,94,179).  visible(mws1gt_5,179). happensAt(walking(mws1gt_5),179).
orientation(mws1gt_1,155,180). coords(mws1gt_1,151,136,180).  visible(mws1gt_1,180). happensAt(walking(mws1gt_1),180). orientation(mws1gt_5,145,180). coords(mws1gt_5,58,95,180).  visible(mws1gt_5,180). happensAt(walking(mws1gt_5),180).
orientation(mws1gt_1,155,181). coords(mws1gt_1,151,136,181).  visible(mws1gt_1,181). happensAt(walking(mws1gt_1),181). orientation(mws1gt_5,145,181). coords(mws1gt_5,58,96,181).  visible(mws1gt_5,181). happensAt(walking(mws1gt_5),181).
orientation(mws1gt_1,155,182). coords(mws1gt_1,150,135,182).  visible(mws1gt_1,182). happensAt(walking(mws1gt_1),182). orientation(mws1gt_5,145,182). coords(mws1gt_5,58,96,182).  visible(mws1gt_5,182). happensAt(walking(mws1gt_5),182).
orientation(mws1gt_1,155,183). coords(mws1gt_1,150,135,183).  visible(mws1gt_1,183). happensAt(walking(mws1gt_1),183). orientation(mws1gt_5,145,183). coords(mws1gt_5,58,97,183).  visible(mws1gt_5,183). happensAt(walking(mws1gt_5),183).
orientation(mws1gt_1,155,184). coords(mws1gt_1,149,136,184).  visible(mws1gt_1,184). happensAt(walking(mws1gt_1),184). orientation(mws1gt_5,145,184). coords(mws1gt_5,58,97,184).  visible(mws1gt_5,184). happensAt(walking(mws1gt_5),184).
orientation(mws1gt_1,155,185). coords(mws1gt_1,149,136,185).  visible(mws1gt_1,185). happensAt(walking(mws1gt_1),185). orientation(mws1gt_5,145,185). coords(mws1gt_5,57,97,185).  visible(mws1gt_5,185). happensAt(walking(mws1gt_5),185).
orientation(mws1gt_1,155,186). coords(mws1gt_1,148,136,186).  visible(mws1gt_1,186). happensAt(walking(mws1gt_1),186). orientation(mws1gt_5,145,186). coords(mws1gt_5,57,97,186).  visible(mws1gt_5,186). happensAt(walking(mws1gt_5),186).
orientation(mws1gt_1,155,187). coords(mws1gt_1,148,136,187).  visible(mws1gt_1,187). happensAt(walking(mws1gt_1),187). orientation(mws1gt_5,145,187). coords(mws1gt_5,57,98,187).  visible(mws1gt_5,187). happensAt(walking(mws1gt_5),187).
orientation(mws1gt_1,155,188). coords(mws1gt_1,147,136,188).  visible(mws1gt_1,188). happensAt(walking(mws1gt_1),188). orientation(mws1gt_5,145,188). coords(mws1gt_5,56,98,188).  visible(mws1gt_5,188). happensAt(walking(mws1gt_5),188).
orientation(mws1gt_1,155,189). coords(mws1gt_1,146,136,189).  visible(mws1gt_1,189). happensAt(walking(mws1gt_1),189). orientation(mws1gt_5,145,189). coords(mws1gt_5,55,99,189).  visible(mws1gt_5,189). happensAt(walking(mws1gt_5),189).
orientation(mws1gt_1,155,190). coords(mws1gt_1,145,137,190).  visible(mws1gt_1,190). happensAt(walking(mws1gt_1),190). orientation(mws1gt_5,145,190). coords(mws1gt_5,55,99,190).  visible(mws1gt_5,190). happensAt(walking(mws1gt_5),190).
#show holdsAt/2.