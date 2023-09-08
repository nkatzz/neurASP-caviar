
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

	orientation(fcgt_2,160,162). coords(fcgt_2,28,207,162).  visible(fcgt_2,162). happensAt(walking(fcgt_2),162). orientation(fcgt_3,159,162). coords(fcgt_3,103,163,162).  visible(fcgt_3,162). happensAt(inactive(fcgt_3),162).
orientation(fcgt_2,166,163). coords(fcgt_2,28,207,163).  visible(fcgt_2,163). happensAt(walking(fcgt_2),163). orientation(fcgt_3,159,163). coords(fcgt_3,103,163,163).  visible(fcgt_3,163). happensAt(inactive(fcgt_3),163).
orientation(fcgt_2,166,164). coords(fcgt_2,29,207,164).  visible(fcgt_2,164). happensAt(walking(fcgt_2),164). orientation(fcgt_3,159,164). coords(fcgt_3,103,163,164).  visible(fcgt_3,164). happensAt(inactive(fcgt_3),164).
orientation(fcgt_2,166,165). coords(fcgt_2,29,207,165).  visible(fcgt_2,165). happensAt(walking(fcgt_2),165). orientation(fcgt_3,159,165). coords(fcgt_3,103,163,165).  visible(fcgt_3,165). happensAt(inactive(fcgt_3),165).
orientation(fcgt_2,166,166). coords(fcgt_2,30,207,166).  visible(fcgt_2,166). happensAt(walking(fcgt_2),166). orientation(fcgt_3,159,166). coords(fcgt_3,103,163,166).  visible(fcgt_3,166). happensAt(inactive(fcgt_3),166).
orientation(fcgt_2,166,167). coords(fcgt_2,30,207,167).  visible(fcgt_2,167). happensAt(walking(fcgt_2),167). orientation(fcgt_3,159,167). coords(fcgt_3,103,163,167).  visible(fcgt_3,167). happensAt(inactive(fcgt_3),167).
orientation(fcgt_2,166,168). coords(fcgt_2,30,207,168).  visible(fcgt_2,168). happensAt(walking(fcgt_2),168). orientation(fcgt_3,159,168). coords(fcgt_3,103,163,168).  visible(fcgt_3,168). happensAt(inactive(fcgt_3),168).
orientation(fcgt_2,166,169). coords(fcgt_2,30,207,169).  visible(fcgt_2,169). happensAt(walking(fcgt_2),169). orientation(fcgt_3,159,169). coords(fcgt_3,103,163,169).  visible(fcgt_3,169). happensAt(inactive(fcgt_3),169).
orientation(fcgt_2,166,170). coords(fcgt_2,31,206,170).  visible(fcgt_2,170). happensAt(walking(fcgt_2),170). orientation(fcgt_3,159,170). coords(fcgt_3,103,163,170).  visible(fcgt_3,170). happensAt(inactive(fcgt_3),170).
orientation(fcgt_2,166,171). coords(fcgt_2,32,206,171).  visible(fcgt_2,171). happensAt(walking(fcgt_2),171). orientation(fcgt_3,159,171). coords(fcgt_3,103,163,171).  visible(fcgt_3,171). happensAt(inactive(fcgt_3),171).
orientation(fcgt_2,166,172). coords(fcgt_2,33,206,172).  visible(fcgt_2,172). happensAt(walking(fcgt_2),172). orientation(fcgt_3,159,172). coords(fcgt_3,103,163,172).  visible(fcgt_3,172). happensAt(inactive(fcgt_3),172).
orientation(fcgt_2,166,173). coords(fcgt_2,33,205,173).  visible(fcgt_2,173). happensAt(walking(fcgt_2),173). orientation(fcgt_3,159,173). coords(fcgt_3,103,163,173).  visible(fcgt_3,173). happensAt(inactive(fcgt_3),173).
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
#show holdsAt/2.