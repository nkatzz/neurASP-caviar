
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

	orientation(ricgt_3,159,159). coords(ricgt_3,209,178,159).  visible(ricgt_3,159). happensAt(walking(ricgt_3),159). orientation(ricgt_4,172,159). coords(ricgt_4,25,139,159).  visible(ricgt_4,159). happensAt(walking(ricgt_4),159).
orientation(ricgt_3,162,160). coords(ricgt_3,211,178,160).  visible(ricgt_3,160). happensAt(walking(ricgt_3),160). orientation(ricgt_4,172,160). coords(ricgt_4,25,139,160).  visible(ricgt_4,160). happensAt(walking(ricgt_4),160).
orientation(ricgt_3,166,161). coords(ricgt_3,211,179,161).  visible(ricgt_3,161). happensAt(walking(ricgt_3),161). orientation(ricgt_4,172,161). coords(ricgt_4,25,139,161).  visible(ricgt_4,161). happensAt(walking(ricgt_4),161).
orientation(ricgt_3,168,162). coords(ricgt_3,212,182,162).  visible(ricgt_3,162). happensAt(walking(ricgt_3),162). orientation(ricgt_4,172,162). coords(ricgt_4,25,139,162).  visible(ricgt_4,162). happensAt(walking(ricgt_4),162).
orientation(ricgt_3,169,163). coords(ricgt_3,214,184,163).  visible(ricgt_3,163). happensAt(walking(ricgt_3),163). orientation(ricgt_4,172,163). coords(ricgt_4,25,139,163).  visible(ricgt_4,163). happensAt(walking(ricgt_4),163).
orientation(ricgt_3,170,164). coords(ricgt_3,216,185,164).  visible(ricgt_3,164). happensAt(walking(ricgt_3),164). orientation(ricgt_4,172,164). coords(ricgt_4,25,139,164).  visible(ricgt_4,164). happensAt(walking(ricgt_4),164).
orientation(ricgt_3,172,165). coords(ricgt_3,220,186,165).  visible(ricgt_3,165). happensAt(walking(ricgt_3),165). orientation(ricgt_4,172,165). coords(ricgt_4,25,140,165).  visible(ricgt_4,165). happensAt(walking(ricgt_4),165).
orientation(ricgt_3,174,166). coords(ricgt_3,223,187,166).  visible(ricgt_3,166). happensAt(walking(ricgt_3),166). orientation(ricgt_4,172,166). coords(ricgt_4,25,140,166).  visible(ricgt_4,166). happensAt(walking(ricgt_4),166).
orientation(ricgt_3,178,167). coords(ricgt_3,228,189,167).  visible(ricgt_3,167). happensAt(walking(ricgt_3),167). orientation(ricgt_4,172,167). coords(ricgt_4,25,140,167).  visible(ricgt_4,167). happensAt(walking(ricgt_4),167).
orientation(ricgt_3,179,168). coords(ricgt_3,230,191,168).  visible(ricgt_3,168). happensAt(walking(ricgt_3),168). orientation(ricgt_4,172,168). coords(ricgt_4,25,140,168).  visible(ricgt_4,168). happensAt(walking(ricgt_4),168).
orientation(ricgt_3,0,169). coords(ricgt_3,233,195,169).  visible(ricgt_3,169). happensAt(walking(ricgt_3),169). orientation(ricgt_4,172,169). coords(ricgt_4,25,140,169).  visible(ricgt_4,169). happensAt(walking(ricgt_4),169).
orientation(ricgt_3,0,170). coords(ricgt_3,235,197,170).  visible(ricgt_3,170). happensAt(walking(ricgt_3),170). orientation(ricgt_4,172,170). coords(ricgt_4,25,141,170).  visible(ricgt_4,170). happensAt(walking(ricgt_4),170).
orientation(ricgt_3,3,171). coords(ricgt_3,236,198,171).  visible(ricgt_3,171). happensAt(walking(ricgt_3),171). orientation(ricgt_4,172,171). coords(ricgt_4,25,141,171).  visible(ricgt_4,171). happensAt(walking(ricgt_4),171).
orientation(ricgt_3,8,172). coords(ricgt_3,238,199,172).  visible(ricgt_3,172). happensAt(walking(ricgt_3),172). orientation(ricgt_4,172,172). coords(ricgt_4,25,141,172).  visible(ricgt_4,172). happensAt(walking(ricgt_4),172).
orientation(ricgt_3,10,173). coords(ricgt_3,238,200,173).  visible(ricgt_3,173). happensAt(walking(ricgt_3),173). orientation(ricgt_4,172,173). coords(ricgt_4,25,141,173).  visible(ricgt_4,173). happensAt(walking(ricgt_4),173).
orientation(ricgt_3,10,174). coords(ricgt_3,239,202,174).  visible(ricgt_3,174). happensAt(walking(ricgt_3),174). orientation(ricgt_4,172,174). coords(ricgt_4,24,141,174).  visible(ricgt_4,174). happensAt(walking(ricgt_4),174).
orientation(ricgt_3,10,175). coords(ricgt_3,240,204,175).  visible(ricgt_3,175). happensAt(walking(ricgt_3),175). orientation(ricgt_4,172,175). coords(ricgt_4,24,141,175).  visible(ricgt_4,175). happensAt(walking(ricgt_4),175).
orientation(ricgt_3,7,176). coords(ricgt_3,241,207,176).  visible(ricgt_3,176). happensAt(walking(ricgt_3),176). orientation(ricgt_4,172,176). coords(ricgt_4,24,141,176).  visible(ricgt_4,176). happensAt(walking(ricgt_4),176).
orientation(ricgt_3,6,177). coords(ricgt_3,242,209,177).  visible(ricgt_3,177). happensAt(walking(ricgt_3),177). orientation(ricgt_4,172,177). coords(ricgt_4,24,141,177).  visible(ricgt_4,177). happensAt(walking(ricgt_4),177).
orientation(ricgt_3,2,178). coords(ricgt_3,243,213,178).  visible(ricgt_3,178). happensAt(walking(ricgt_3),178). orientation(ricgt_4,172,178). coords(ricgt_4,24,141,178).  visible(ricgt_4,178). happensAt(walking(ricgt_4),178).
orientation(ricgt_3,7,179). coords(ricgt_3,247,217,179).  visible(ricgt_3,179). happensAt(walking(ricgt_3),179). orientation(ricgt_4,172,179). coords(ricgt_4,24,141,179).  visible(ricgt_4,179). happensAt(walking(ricgt_4),179).
orientation(ricgt_3,3,180). coords(ricgt_3,250,220,180).  visible(ricgt_3,180). happensAt(walking(ricgt_3),180). orientation(ricgt_4,172,180). coords(ricgt_4,24,141,180).  visible(ricgt_4,180). happensAt(walking(ricgt_4),180).
orientation(ricgt_3,9,181). coords(ricgt_3,253,223,181).  visible(ricgt_3,181). happensAt(walking(ricgt_3),181). orientation(ricgt_4,172,181). coords(ricgt_4,24,141,181).  visible(ricgt_4,181). happensAt(walking(ricgt_4),181).
orientation(ricgt_3,6,182). coords(ricgt_3,256,224,182).  visible(ricgt_3,182). happensAt(walking(ricgt_3),182). orientation(ricgt_4,172,182). coords(ricgt_4,24,141,182).  visible(ricgt_4,182). happensAt(walking(ricgt_4),182).
orientation(ricgt_3,9,183). coords(ricgt_3,258,225,183).  visible(ricgt_3,183). happensAt(walking(ricgt_3),183). orientation(ricgt_4,172,183). coords(ricgt_4,24,142,183).  visible(ricgt_4,183). happensAt(walking(ricgt_4),183).
#show holdsAt/2.