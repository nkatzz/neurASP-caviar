
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

	orientation(lb2gt_0,141,161). coords(lb2gt_0,185,116,161).  visible(lb2gt_0,161). happensAt(active(lb2gt_0),161). orientation(lb2gt_3,172,161). coords(lb2gt_3,5,67,161).  visible(lb2gt_3,161). happensAt(walking(lb2gt_3),161).
orientation(lb2gt_0,141,162). coords(lb2gt_0,185,116,162).  visible(lb2gt_0,162). happensAt(active(lb2gt_0),162). orientation(lb2gt_3,174,162). coords(lb2gt_3,5,68,162).  visible(lb2gt_3,162). happensAt(walking(lb2gt_3),162).
orientation(lb2gt_0,141,163). coords(lb2gt_0,185,116,163).  visible(lb2gt_0,163). happensAt(active(lb2gt_0),163). orientation(lb2gt_3,174,163). coords(lb2gt_3,5,68,163).  visible(lb2gt_3,163). happensAt(walking(lb2gt_3),163).
orientation(lb2gt_0,141,164). coords(lb2gt_0,186,116,164).  visible(lb2gt_0,164). happensAt(active(lb2gt_0),164). orientation(lb2gt_3,176,164). coords(lb2gt_3,5,68,164).  visible(lb2gt_3,164). happensAt(walking(lb2gt_3),164).
orientation(lb2gt_0,141,165). coords(lb2gt_0,186,116,165).  visible(lb2gt_0,165). happensAt(active(lb2gt_0),165). orientation(lb2gt_3,174,165). coords(lb2gt_3,5,68,165).  visible(lb2gt_3,165). happensAt(walking(lb2gt_3),165).
orientation(lb2gt_0,141,166). coords(lb2gt_0,186,116,166).  visible(lb2gt_0,166). happensAt(active(lb2gt_0),166). orientation(lb2gt_3,177,166). coords(lb2gt_3,5,68,166).  visible(lb2gt_3,166). happensAt(walking(lb2gt_3),166).
orientation(lb2gt_0,141,167). coords(lb2gt_0,186,116,167).  visible(lb2gt_0,167). happensAt(active(lb2gt_0),167). orientation(lb2gt_3,174,167). coords(lb2gt_3,5,68,167).  visible(lb2gt_3,167). happensAt(walking(lb2gt_3),167).
orientation(lb2gt_0,141,168). coords(lb2gt_0,187,117,168).  visible(lb2gt_0,168). happensAt(active(lb2gt_0),168). orientation(lb2gt_3,0,168). coords(lb2gt_3,5,68,168).  visible(lb2gt_3,168). happensAt(walking(lb2gt_3),168).
orientation(lb2gt_0,141,169). coords(lb2gt_0,187,117,169).  visible(lb2gt_0,169). happensAt(active(lb2gt_0),169). orientation(lb2gt_3,175,169). coords(lb2gt_3,5,68,169).  visible(lb2gt_3,169). happensAt(walking(lb2gt_3),169).
orientation(lb2gt_0,141,170). coords(lb2gt_0,187,117,170).  visible(lb2gt_0,170). happensAt(active(lb2gt_0),170). orientation(lb2gt_3,178,170). coords(lb2gt_3,5,68,170).  visible(lb2gt_3,170). happensAt(walking(lb2gt_3),170).
orientation(lb2gt_0,141,171). coords(lb2gt_0,187,117,171).  visible(lb2gt_0,171). happensAt(active(lb2gt_0),171). orientation(lb2gt_3,176,171). coords(lb2gt_3,5,68,171).  visible(lb2gt_3,171). happensAt(walking(lb2gt_3),171).
orientation(lb2gt_0,141,172). coords(lb2gt_0,187,117,172).  visible(lb2gt_0,172). happensAt(active(lb2gt_0),172). orientation(lb2gt_3,177,172). coords(lb2gt_3,4,69,172).  visible(lb2gt_3,172). happensAt(walking(lb2gt_3),172).
orientation(lb2gt_0,141,173). coords(lb2gt_0,188,118,173).  visible(lb2gt_0,173). happensAt(active(lb2gt_0),173). orientation(lb2gt_3,175,173). coords(lb2gt_3,4,69,173).  visible(lb2gt_3,173). happensAt(walking(lb2gt_3),173).
orientation(lb2gt_0,141,174). coords(lb2gt_0,188,118,174).  visible(lb2gt_0,174). happensAt(active(lb2gt_0),174). orientation(lb2gt_3,174,174). coords(lb2gt_3,4,69,174).  visible(lb2gt_3,174). happensAt(walking(lb2gt_3),174).
orientation(lb2gt_0,141,175). coords(lb2gt_0,188,118,175).  visible(lb2gt_0,175). happensAt(active(lb2gt_0),175). orientation(lb2gt_3,2,175). coords(lb2gt_3,4,69,175).  visible(lb2gt_3,175). happensAt(walking(lb2gt_3),175).
orientation(lb2gt_0,141,176). coords(lb2gt_0,189,119,176).  visible(lb2gt_0,176). happensAt(active(lb2gt_0),176). orientation(lb2gt_3,5,176). coords(lb2gt_3,4,69,176).  visible(lb2gt_3,176). happensAt(walking(lb2gt_3),176).
orientation(lb2gt_0,141,177). coords(lb2gt_0,189,120,177).  visible(lb2gt_0,177). happensAt(active(lb2gt_0),177). orientation(lb2gt_3,1,177). coords(lb2gt_3,4,69,177).  visible(lb2gt_3,177). happensAt(walking(lb2gt_3),177).
orientation(lb2gt_0,141,178). coords(lb2gt_0,189,121,178).  visible(lb2gt_0,178). happensAt(active(lb2gt_0),178). orientation(lb2gt_3,3,178). coords(lb2gt_3,4,69,178).  visible(lb2gt_3,178). happensAt(walking(lb2gt_3),178).
orientation(lb2gt_0,141,179). coords(lb2gt_0,191,122,179).  visible(lb2gt_0,179). happensAt(active(lb2gt_0),179). orientation(lb2gt_3,5,179). coords(lb2gt_3,4,69,179).  visible(lb2gt_3,179). happensAt(walking(lb2gt_3),179).
orientation(lb2gt_0,141,180). coords(lb2gt_0,192,124,180).  visible(lb2gt_0,180). happensAt(walking(lb2gt_0),180). orientation(lb2gt_3,1,180). coords(lb2gt_3,4,71,180).  visible(lb2gt_3,180). happensAt(walking(lb2gt_3),180).
orientation(lb2gt_0,141,181). coords(lb2gt_0,194,125,181).  visible(lb2gt_0,181). happensAt(walking(lb2gt_0),181). orientation(lb2gt_3,7,181). coords(lb2gt_3,4,71,181).  visible(lb2gt_3,181). happensAt(walking(lb2gt_3),181).
orientation(lb2gt_0,141,182). coords(lb2gt_0,196,127,182).  visible(lb2gt_0,182). happensAt(walking(lb2gt_0),182). orientation(lb2gt_3,7,182). coords(lb2gt_3,4,71,182).  visible(lb2gt_3,182). happensAt(walking(lb2gt_3),182).
orientation(lb2gt_0,141,183). coords(lb2gt_0,198,128,183).  visible(lb2gt_0,183). happensAt(walking(lb2gt_0),183). orientation(lb2gt_3,3,183). coords(lb2gt_3,4,71,183).  visible(lb2gt_3,183). happensAt(walking(lb2gt_3),183).
orientation(lb2gt_0,141,184). coords(lb2gt_0,199,130,184).  visible(lb2gt_0,184). happensAt(walking(lb2gt_0),184). orientation(lb2gt_3,2,184). coords(lb2gt_3,3,73,184).  visible(lb2gt_3,184). happensAt(walking(lb2gt_3),184).
orientation(lb2gt_0,141,185). coords(lb2gt_0,201,132,185).  visible(lb2gt_0,185). happensAt(walking(lb2gt_0),185). orientation(lb2gt_3,174,185). coords(lb2gt_3,3,73,185).  happensAt(disappear(lb2gt_3),185). happensAt(walking(lb2gt_3),185).
#show holdsAt/2.