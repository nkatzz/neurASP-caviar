
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

	orientation(fra2gt_3,165,163). coords(fra2gt_3,36,198,163).  visible(fra2gt_3,163). happensAt(walking(fra2gt_3),163). orientation(fra2gt_5,150,163). coords(fra2gt_5,29,117,163).  visible(fra2gt_5,163). happensAt(walking(fra2gt_5),163).
orientation(fra2gt_3,165,164). coords(fra2gt_3,36,197,164).  visible(fra2gt_3,164). happensAt(walking(fra2gt_3),164). orientation(fra2gt_5,150,164). coords(fra2gt_5,29,117,164).  visible(fra2gt_5,164). happensAt(walking(fra2gt_5),164).
orientation(fra2gt_3,165,165). coords(fra2gt_3,36,197,165).  visible(fra2gt_3,165). happensAt(walking(fra2gt_3),165). orientation(fra2gt_5,150,165). coords(fra2gt_5,29,117,165).  visible(fra2gt_5,165). happensAt(walking(fra2gt_5),165).
orientation(fra2gt_3,165,166). coords(fra2gt_3,36,197,166).  visible(fra2gt_3,166). happensAt(walking(fra2gt_3),166). orientation(fra2gt_5,150,166). coords(fra2gt_5,29,117,166).  visible(fra2gt_5,166). happensAt(walking(fra2gt_5),166).
orientation(fra2gt_3,161,167). coords(fra2gt_3,36,197,167).  visible(fra2gt_3,167). happensAt(walking(fra2gt_3),167). orientation(fra2gt_5,150,167). coords(fra2gt_5,29,117,167).  visible(fra2gt_5,167). happensAt(walking(fra2gt_5),167).
orientation(fra2gt_3,161,168). coords(fra2gt_3,37,196,168).  visible(fra2gt_3,168). happensAt(walking(fra2gt_3),168). orientation(fra2gt_5,150,168). coords(fra2gt_5,29,117,168).  visible(fra2gt_5,168). happensAt(walking(fra2gt_5),168).
orientation(fra2gt_3,161,169). coords(fra2gt_3,37,196,169).  visible(fra2gt_3,169). happensAt(walking(fra2gt_3),169). orientation(fra2gt_5,150,169). coords(fra2gt_5,29,117,169).  visible(fra2gt_5,169). happensAt(walking(fra2gt_5),169).
orientation(fra2gt_3,161,170). coords(fra2gt_3,37,196,170).  visible(fra2gt_3,170). happensAt(walking(fra2gt_3),170). orientation(fra2gt_5,150,170). coords(fra2gt_5,29,117,170).  visible(fra2gt_5,170). happensAt(walking(fra2gt_5),170).
orientation(fra2gt_3,161,171). coords(fra2gt_3,37,195,171).  visible(fra2gt_3,171). happensAt(walking(fra2gt_3),171). orientation(fra2gt_5,150,171). coords(fra2gt_5,29,117,171).  visible(fra2gt_5,171). happensAt(walking(fra2gt_5),171).
orientation(fra2gt_3,161,172). coords(fra2gt_3,37,194,172).  visible(fra2gt_3,172). happensAt(walking(fra2gt_3),172). orientation(fra2gt_5,150,172). coords(fra2gt_5,29,117,172).  visible(fra2gt_5,172). happensAt(walking(fra2gt_5),172).
orientation(fra2gt_3,161,173). coords(fra2gt_3,37,194,173).  visible(fra2gt_3,173). happensAt(walking(fra2gt_3),173). orientation(fra2gt_5,150,173). coords(fra2gt_5,29,117,173).  visible(fra2gt_5,173). happensAt(walking(fra2gt_5),173).
orientation(fra2gt_3,161,174). coords(fra2gt_3,37,194,174).  visible(fra2gt_3,174). happensAt(walking(fra2gt_3),174). orientation(fra2gt_5,150,174). coords(fra2gt_5,29,117,174).  visible(fra2gt_5,174). happensAt(walking(fra2gt_5),174).
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
#show holdsAt/2.