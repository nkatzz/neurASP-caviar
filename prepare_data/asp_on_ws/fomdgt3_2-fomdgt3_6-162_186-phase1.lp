
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

	orientation(fomdgt3_2,160,162). coords(fomdgt3_2,254,207,162).  visible(fomdgt3_2,162). happensAt(walking(fomdgt3_2),162). orientation(fomdgt3_6,110,162). coords(fomdgt3_6,240,28,162).  visible(fomdgt3_6,162). happensAt(walking(fomdgt3_6),162).
orientation(fomdgt3_2,160,163). coords(fomdgt3_2,254,203,163).  visible(fomdgt3_2,163). happensAt(walking(fomdgt3_2),163). orientation(fomdgt3_6,110,163). coords(fomdgt3_6,240,28,163).  visible(fomdgt3_6,163). happensAt(walking(fomdgt3_6),163).
orientation(fomdgt3_2,160,164). coords(fomdgt3_2,253,200,164).  visible(fomdgt3_2,164). happensAt(walking(fomdgt3_2),164). orientation(fomdgt3_6,110,164). coords(fomdgt3_6,240,31,164).  visible(fomdgt3_6,164). happensAt(walking(fomdgt3_6),164).
orientation(fomdgt3_2,160,165). coords(fomdgt3_2,252,199,165).  visible(fomdgt3_2,165). happensAt(walking(fomdgt3_2),165). orientation(fomdgt3_6,110,165). coords(fomdgt3_6,240,31,165).  visible(fomdgt3_6,165). happensAt(walking(fomdgt3_6),165).
orientation(fomdgt3_2,160,166). coords(fomdgt3_2,252,196,166).  visible(fomdgt3_2,166). happensAt(walking(fomdgt3_2),166). orientation(fomdgt3_6,110,166). coords(fomdgt3_6,236,33,166).  visible(fomdgt3_6,166). happensAt(walking(fomdgt3_6),166).
orientation(fomdgt3_2,160,167). coords(fomdgt3_2,251,195,167).  visible(fomdgt3_2,167). happensAt(walking(fomdgt3_2),167). orientation(fomdgt3_6,110,167). coords(fomdgt3_6,236,33,167).  visible(fomdgt3_6,167). happensAt(walking(fomdgt3_6),167).
orientation(fomdgt3_2,160,168). coords(fomdgt3_2,251,195,168).  visible(fomdgt3_2,168). happensAt(walking(fomdgt3_2),168). orientation(fomdgt3_6,110,168). coords(fomdgt3_6,233,36,168).  visible(fomdgt3_6,168). happensAt(walking(fomdgt3_6),168).
orientation(fomdgt3_2,160,169). coords(fomdgt3_2,251,195,169).  visible(fomdgt3_2,169). happensAt(walking(fomdgt3_2),169). orientation(fomdgt3_6,110,169). coords(fomdgt3_6,233,36,169).  visible(fomdgt3_6,169). happensAt(walking(fomdgt3_6),169).
orientation(fomdgt3_2,160,170). coords(fomdgt3_2,250,193,170).  visible(fomdgt3_2,170). happensAt(walking(fomdgt3_2),170). orientation(fomdgt3_6,110,170). coords(fomdgt3_6,233,36,170).  visible(fomdgt3_6,170). happensAt(walking(fomdgt3_6),170).
orientation(fomdgt3_2,160,171). coords(fomdgt3_2,249,193,171).  visible(fomdgt3_2,171). happensAt(walking(fomdgt3_2),171). orientation(fomdgt3_6,110,171). coords(fomdgt3_6,231,37,171).  visible(fomdgt3_6,171). happensAt(walking(fomdgt3_6),171).
orientation(fomdgt3_2,160,172). coords(fomdgt3_2,249,192,172).  visible(fomdgt3_2,172). happensAt(walking(fomdgt3_2),172). orientation(fomdgt3_6,110,172). coords(fomdgt3_6,231,37,172).  visible(fomdgt3_6,172). happensAt(walking(fomdgt3_6),172).
orientation(fomdgt3_2,160,173). coords(fomdgt3_2,248,189,173).  visible(fomdgt3_2,173). happensAt(walking(fomdgt3_2),173). orientation(fomdgt3_6,110,173). coords(fomdgt3_6,230,41,173).  visible(fomdgt3_6,173). happensAt(walking(fomdgt3_6),173).
orientation(fomdgt3_2,160,174). coords(fomdgt3_2,247,187,174).  visible(fomdgt3_2,174). happensAt(walking(fomdgt3_2),174). orientation(fomdgt3_6,130,174). coords(fomdgt3_6,230,45,174).  visible(fomdgt3_6,174). happensAt(walking(fomdgt3_6),174).
orientation(fomdgt3_2,160,175). coords(fomdgt3_2,245,184,175).  visible(fomdgt3_2,175). happensAt(walking(fomdgt3_2),175). orientation(fomdgt3_6,130,175). coords(fomdgt3_6,229,46,175).  visible(fomdgt3_6,175). happensAt(walking(fomdgt3_6),175).
orientation(fomdgt3_2,160,176). coords(fomdgt3_2,245,184,176).  visible(fomdgt3_2,176). happensAt(walking(fomdgt3_2),176). orientation(fomdgt3_6,130,176). coords(fomdgt3_6,229,46,176).  visible(fomdgt3_6,176). happensAt(walking(fomdgt3_6),176).
orientation(fomdgt3_2,160,177). coords(fomdgt3_2,243,179,177).  visible(fomdgt3_2,177). happensAt(walking(fomdgt3_2),177). orientation(fomdgt3_6,130,177). coords(fomdgt3_6,228,49,177).  visible(fomdgt3_6,177). happensAt(walking(fomdgt3_6),177).
orientation(fomdgt3_2,160,178). coords(fomdgt3_2,242,177,178).  visible(fomdgt3_2,178). happensAt(walking(fomdgt3_2),178). orientation(fomdgt3_6,130,178). coords(fomdgt3_6,228,49,178).  visible(fomdgt3_6,178). happensAt(walking(fomdgt3_6),178).
orientation(fomdgt3_2,160,179). coords(fomdgt3_2,242,176,179).  visible(fomdgt3_2,179). happensAt(walking(fomdgt3_2),179). orientation(fomdgt3_6,130,179). coords(fomdgt3_6,227,50,179).  visible(fomdgt3_6,179). happensAt(walking(fomdgt3_6),179).
orientation(fomdgt3_2,160,180). coords(fomdgt3_2,242,176,180).  visible(fomdgt3_2,180). happensAt(walking(fomdgt3_2),180). orientation(fomdgt3_6,130,180). coords(fomdgt3_6,227,52,180).  visible(fomdgt3_6,180). happensAt(walking(fomdgt3_6),180).
orientation(fomdgt3_2,160,181). coords(fomdgt3_2,242,176,181).  visible(fomdgt3_2,181). happensAt(walking(fomdgt3_2),181). orientation(fomdgt3_6,130,181). coords(fomdgt3_6,227,52,181).  visible(fomdgt3_6,181). happensAt(walking(fomdgt3_6),181).
orientation(fomdgt3_2,160,182). coords(fomdgt3_2,239,172,182).  visible(fomdgt3_2,182). happensAt(walking(fomdgt3_2),182). orientation(fomdgt3_6,130,182). coords(fomdgt3_6,223,52,182).  visible(fomdgt3_6,182). happensAt(walking(fomdgt3_6),182).
orientation(fomdgt3_2,160,183). coords(fomdgt3_2,239,171,183).  visible(fomdgt3_2,183). happensAt(walking(fomdgt3_2),183). orientation(fomdgt3_6,130,183). coords(fomdgt3_6,223,52,183).  visible(fomdgt3_6,183). happensAt(walking(fomdgt3_6),183).
orientation(fomdgt3_2,160,184). coords(fomdgt3_2,239,170,184).  visible(fomdgt3_2,184). happensAt(walking(fomdgt3_2),184). orientation(fomdgt3_6,130,184). coords(fomdgt3_6,223,56,184).  visible(fomdgt3_6,184). happensAt(walking(fomdgt3_6),184).
orientation(fomdgt3_2,130,185). coords(fomdgt3_2,239,168,185).  visible(fomdgt3_2,185). happensAt(walking(fomdgt3_2),185). orientation(fomdgt3_6,130,185). coords(fomdgt3_6,220,60,185).  visible(fomdgt3_6,185). happensAt(walking(fomdgt3_6),185).
orientation(fomdgt3_2,130,186). coords(fomdgt3_2,239,168,186).  visible(fomdgt3_2,186). happensAt(walking(fomdgt3_2),186). orientation(fomdgt3_6,130,186). coords(fomdgt3_6,220,63,186).  visible(fomdgt3_6,186). happensAt(walking(fomdgt3_6),186).
#show holdsAt/2.