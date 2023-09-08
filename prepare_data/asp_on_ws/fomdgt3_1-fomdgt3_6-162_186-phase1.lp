
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

	orientation(fomdgt3_1,150,162). coords(fomdgt3_1,97,108,162).  visible(fomdgt3_1,162). happensAt(walking(fomdgt3_1),162). orientation(fomdgt3_6,110,162). coords(fomdgt3_6,240,28,162).  visible(fomdgt3_6,162). happensAt(walking(fomdgt3_6),162).
orientation(fomdgt3_1,150,163). coords(fomdgt3_1,97,108,163).  visible(fomdgt3_1,163). happensAt(walking(fomdgt3_1),163). orientation(fomdgt3_6,110,163). coords(fomdgt3_6,240,28,163).  visible(fomdgt3_6,163). happensAt(walking(fomdgt3_6),163).
orientation(fomdgt3_1,150,164). coords(fomdgt3_1,97,108,164).  visible(fomdgt3_1,164). happensAt(walking(fomdgt3_1),164). orientation(fomdgt3_6,110,164). coords(fomdgt3_6,240,31,164).  visible(fomdgt3_6,164). happensAt(walking(fomdgt3_6),164).
orientation(fomdgt3_1,150,165). coords(fomdgt3_1,97,108,165).  visible(fomdgt3_1,165). happensAt(walking(fomdgt3_1),165). orientation(fomdgt3_6,110,165). coords(fomdgt3_6,240,31,165).  visible(fomdgt3_6,165). happensAt(walking(fomdgt3_6),165).
orientation(fomdgt3_1,150,166). coords(fomdgt3_1,96,108,166).  visible(fomdgt3_1,166). happensAt(walking(fomdgt3_1),166). orientation(fomdgt3_6,110,166). coords(fomdgt3_6,236,33,166).  visible(fomdgt3_6,166). happensAt(walking(fomdgt3_6),166).
orientation(fomdgt3_1,150,167). coords(fomdgt3_1,96,108,167).  visible(fomdgt3_1,167). happensAt(walking(fomdgt3_1),167). orientation(fomdgt3_6,110,167). coords(fomdgt3_6,236,33,167).  visible(fomdgt3_6,167). happensAt(walking(fomdgt3_6),167).
orientation(fomdgt3_1,150,168). coords(fomdgt3_1,96,108,168).  visible(fomdgt3_1,168). happensAt(walking(fomdgt3_1),168). orientation(fomdgt3_6,110,168). coords(fomdgt3_6,233,36,168).  visible(fomdgt3_6,168). happensAt(walking(fomdgt3_6),168).
orientation(fomdgt3_1,150,169). coords(fomdgt3_1,92,107,169).  visible(fomdgt3_1,169). happensAt(walking(fomdgt3_1),169). orientation(fomdgt3_6,110,169). coords(fomdgt3_6,233,36,169).  visible(fomdgt3_6,169). happensAt(walking(fomdgt3_6),169).
orientation(fomdgt3_1,150,170). coords(fomdgt3_1,91,107,170).  visible(fomdgt3_1,170). happensAt(walking(fomdgt3_1),170). orientation(fomdgt3_6,110,170). coords(fomdgt3_6,233,36,170).  visible(fomdgt3_6,170). happensAt(walking(fomdgt3_6),170).
orientation(fomdgt3_1,150,171). coords(fomdgt3_1,90,106,171).  visible(fomdgt3_1,171). happensAt(walking(fomdgt3_1),171). orientation(fomdgt3_6,110,171). coords(fomdgt3_6,231,37,171).  visible(fomdgt3_6,171). happensAt(walking(fomdgt3_6),171).
orientation(fomdgt3_1,150,172). coords(fomdgt3_1,90,106,172).  visible(fomdgt3_1,172). happensAt(walking(fomdgt3_1),172). orientation(fomdgt3_6,110,172). coords(fomdgt3_6,231,37,172).  visible(fomdgt3_6,172). happensAt(walking(fomdgt3_6),172).
orientation(fomdgt3_1,150,173). coords(fomdgt3_1,89,105,173).  visible(fomdgt3_1,173). happensAt(walking(fomdgt3_1),173). orientation(fomdgt3_6,110,173). coords(fomdgt3_6,230,41,173).  visible(fomdgt3_6,173). happensAt(walking(fomdgt3_6),173).
orientation(fomdgt3_1,150,174). coords(fomdgt3_1,88,105,174).  visible(fomdgt3_1,174). happensAt(walking(fomdgt3_1),174). orientation(fomdgt3_6,130,174). coords(fomdgt3_6,230,45,174).  visible(fomdgt3_6,174). happensAt(walking(fomdgt3_6),174).
orientation(fomdgt3_1,150,175). coords(fomdgt3_1,89,105,175).  visible(fomdgt3_1,175). happensAt(walking(fomdgt3_1),175). orientation(fomdgt3_6,130,175). coords(fomdgt3_6,229,46,175).  visible(fomdgt3_6,175). happensAt(walking(fomdgt3_6),175).
orientation(fomdgt3_1,150,176). coords(fomdgt3_1,89,105,176).  visible(fomdgt3_1,176). happensAt(walking(fomdgt3_1),176). orientation(fomdgt3_6,130,176). coords(fomdgt3_6,229,46,176).  visible(fomdgt3_6,176). happensAt(walking(fomdgt3_6),176).
orientation(fomdgt3_1,150,177). coords(fomdgt3_1,89,105,177).  visible(fomdgt3_1,177). happensAt(walking(fomdgt3_1),177). orientation(fomdgt3_6,130,177). coords(fomdgt3_6,228,49,177).  visible(fomdgt3_6,177). happensAt(walking(fomdgt3_6),177).
orientation(fomdgt3_1,150,178). coords(fomdgt3_1,88,105,178).  visible(fomdgt3_1,178). happensAt(walking(fomdgt3_1),178). orientation(fomdgt3_6,130,178). coords(fomdgt3_6,228,49,178).  visible(fomdgt3_6,178). happensAt(walking(fomdgt3_6),178).
orientation(fomdgt3_1,150,179). coords(fomdgt3_1,89,105,179).  visible(fomdgt3_1,179). happensAt(walking(fomdgt3_1),179). orientation(fomdgt3_6,130,179). coords(fomdgt3_6,227,50,179).  visible(fomdgt3_6,179). happensAt(walking(fomdgt3_6),179).
orientation(fomdgt3_1,150,180). coords(fomdgt3_1,87,105,180).  visible(fomdgt3_1,180). happensAt(walking(fomdgt3_1),180). orientation(fomdgt3_6,130,180). coords(fomdgt3_6,227,52,180).  visible(fomdgt3_6,180). happensAt(walking(fomdgt3_6),180).
orientation(fomdgt3_1,150,181). coords(fomdgt3_1,86,103,181).  visible(fomdgt3_1,181). happensAt(walking(fomdgt3_1),181). orientation(fomdgt3_6,130,181). coords(fomdgt3_6,227,52,181).  visible(fomdgt3_6,181). happensAt(walking(fomdgt3_6),181).
orientation(fomdgt3_1,150,182). coords(fomdgt3_1,86,103,182).  visible(fomdgt3_1,182). happensAt(walking(fomdgt3_1),182). orientation(fomdgt3_6,130,182). coords(fomdgt3_6,223,52,182).  visible(fomdgt3_6,182). happensAt(walking(fomdgt3_6),182).
orientation(fomdgt3_1,150,183). coords(fomdgt3_1,86,103,183).  visible(fomdgt3_1,183). happensAt(walking(fomdgt3_1),183). orientation(fomdgt3_6,130,183). coords(fomdgt3_6,223,52,183).  visible(fomdgt3_6,183). happensAt(walking(fomdgt3_6),183).
orientation(fomdgt3_1,150,184). coords(fomdgt3_1,83,102,184).  visible(fomdgt3_1,184). happensAt(walking(fomdgt3_1),184). orientation(fomdgt3_6,130,184). coords(fomdgt3_6,223,56,184).  visible(fomdgt3_6,184). happensAt(walking(fomdgt3_6),184).
orientation(fomdgt3_1,150,185). coords(fomdgt3_1,83,102,185).  visible(fomdgt3_1,185). happensAt(walking(fomdgt3_1),185). orientation(fomdgt3_6,130,185). coords(fomdgt3_6,220,60,185).  visible(fomdgt3_6,185). happensAt(walking(fomdgt3_6),185).
orientation(fomdgt3_1,150,186). coords(fomdgt3_1,82,102,186).  visible(fomdgt3_1,186). happensAt(walking(fomdgt3_1),186). orientation(fomdgt3_6,130,186). coords(fomdgt3_6,220,63,186).  visible(fomdgt3_6,186). happensAt(walking(fomdgt3_6),186).
#show holdsAt/2.