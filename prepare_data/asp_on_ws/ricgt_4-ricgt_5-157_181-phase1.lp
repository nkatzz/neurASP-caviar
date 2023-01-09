
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

	orientation(ricgt_4,172,157). coords(ricgt_4,25,138,157).  visible(ricgt_4,157). happensAt(walking(ricgt_4),157). orientation(ricgt_5,0,157). coords(ricgt_5,27,132,157).  visible(ricgt_5,157). happensAt(walking(ricgt_5),157).
orientation(ricgt_4,172,158). coords(ricgt_4,25,138,158).  visible(ricgt_4,158). happensAt(walking(ricgt_4),158). orientation(ricgt_5,0,158). coords(ricgt_5,27,132,158).  visible(ricgt_5,158). happensAt(walking(ricgt_5),158).
orientation(ricgt_4,172,159). coords(ricgt_4,25,139,159).  visible(ricgt_4,159). happensAt(walking(ricgt_4),159). orientation(ricgt_5,0,159). coords(ricgt_5,27,132,159).  visible(ricgt_5,159). happensAt(walking(ricgt_5),159).
orientation(ricgt_4,172,160). coords(ricgt_4,25,139,160).  visible(ricgt_4,160). happensAt(walking(ricgt_4),160). orientation(ricgt_5,0,160). coords(ricgt_5,27,133,160).  visible(ricgt_5,160). happensAt(walking(ricgt_5),160).
orientation(ricgt_4,172,161). coords(ricgt_4,25,139,161).  visible(ricgt_4,161). happensAt(walking(ricgt_4),161). orientation(ricgt_5,0,161). coords(ricgt_5,27,133,161).  visible(ricgt_5,161). happensAt(walking(ricgt_5),161).
orientation(ricgt_4,172,162). coords(ricgt_4,25,139,162).  visible(ricgt_4,162). happensAt(walking(ricgt_4),162). orientation(ricgt_5,0,162). coords(ricgt_5,27,133,162).  visible(ricgt_5,162). happensAt(walking(ricgt_5),162).
orientation(ricgt_4,172,163). coords(ricgt_4,25,139,163).  visible(ricgt_4,163). happensAt(walking(ricgt_4),163). orientation(ricgt_5,0,163). coords(ricgt_5,27,133,163).  visible(ricgt_5,163). happensAt(walking(ricgt_5),163).
orientation(ricgt_4,172,164). coords(ricgt_4,25,139,164).  visible(ricgt_4,164). happensAt(walking(ricgt_4),164). orientation(ricgt_5,0,164). coords(ricgt_5,27,133,164).  visible(ricgt_5,164). happensAt(walking(ricgt_5),164).
orientation(ricgt_4,172,165). coords(ricgt_4,25,140,165).  visible(ricgt_4,165). happensAt(walking(ricgt_4),165). orientation(ricgt_5,0,165). coords(ricgt_5,27,133,165).  visible(ricgt_5,165). happensAt(walking(ricgt_5),165).
orientation(ricgt_4,172,166). coords(ricgt_4,25,140,166).  visible(ricgt_4,166). happensAt(walking(ricgt_4),166). orientation(ricgt_5,0,166). coords(ricgt_5,27,133,166).  visible(ricgt_5,166). happensAt(walking(ricgt_5),166).
orientation(ricgt_4,172,167). coords(ricgt_4,25,140,167).  visible(ricgt_4,167). happensAt(walking(ricgt_4),167). orientation(ricgt_5,0,167). coords(ricgt_5,27,133,167).  visible(ricgt_5,167). happensAt(walking(ricgt_5),167).
orientation(ricgt_4,172,168). coords(ricgt_4,25,140,168).  visible(ricgt_4,168). happensAt(walking(ricgt_4),168). orientation(ricgt_5,0,168). coords(ricgt_5,26,134,168).  visible(ricgt_5,168). happensAt(walking(ricgt_5),168).
orientation(ricgt_4,172,169). coords(ricgt_4,25,140,169).  visible(ricgt_4,169). happensAt(walking(ricgt_4),169). orientation(ricgt_5,0,169). coords(ricgt_5,26,134,169).  visible(ricgt_5,169). happensAt(walking(ricgt_5),169).
orientation(ricgt_4,172,170). coords(ricgt_4,25,141,170).  visible(ricgt_4,170). happensAt(walking(ricgt_4),170). orientation(ricgt_5,0,170). coords(ricgt_5,26,134,170).  visible(ricgt_5,170). happensAt(walking(ricgt_5),170).
orientation(ricgt_4,172,171). coords(ricgt_4,25,141,171).  visible(ricgt_4,171). happensAt(walking(ricgt_4),171). orientation(ricgt_5,0,171). coords(ricgt_5,26,135,171).  visible(ricgt_5,171). happensAt(walking(ricgt_5),171).
orientation(ricgt_4,172,172). coords(ricgt_4,25,141,172).  visible(ricgt_4,172). happensAt(walking(ricgt_4),172). orientation(ricgt_5,0,172). coords(ricgt_5,26,135,172).  visible(ricgt_5,172). happensAt(walking(ricgt_5),172).
orientation(ricgt_4,172,173). coords(ricgt_4,25,141,173).  visible(ricgt_4,173). happensAt(walking(ricgt_4),173). orientation(ricgt_5,0,173). coords(ricgt_5,26,135,173).  visible(ricgt_5,173). happensAt(walking(ricgt_5),173).
orientation(ricgt_4,172,174). coords(ricgt_4,24,141,174).  visible(ricgt_4,174). happensAt(walking(ricgt_4),174). orientation(ricgt_5,0,174). coords(ricgt_5,26,136,174).  visible(ricgt_5,174). happensAt(walking(ricgt_5),174).
orientation(ricgt_4,172,175). coords(ricgt_4,24,141,175).  visible(ricgt_4,175). happensAt(walking(ricgt_4),175). orientation(ricgt_5,0,175). coords(ricgt_5,26,137,175).  visible(ricgt_5,175). happensAt(walking(ricgt_5),175).
orientation(ricgt_4,172,176). coords(ricgt_4,24,141,176).  visible(ricgt_4,176). happensAt(walking(ricgt_4),176). orientation(ricgt_5,0,176). coords(ricgt_5,26,138,176).  visible(ricgt_5,176). happensAt(walking(ricgt_5),176).
orientation(ricgt_4,172,177). coords(ricgt_4,24,141,177).  visible(ricgt_4,177). happensAt(walking(ricgt_4),177). orientation(ricgt_5,0,177). coords(ricgt_5,26,138,177).  visible(ricgt_5,177). happensAt(walking(ricgt_5),177).
orientation(ricgt_4,172,178). coords(ricgt_4,24,141,178).  visible(ricgt_4,178). happensAt(walking(ricgt_4),178). orientation(ricgt_5,0,178). coords(ricgt_5,26,138,178).  visible(ricgt_5,178). happensAt(walking(ricgt_5),178).
orientation(ricgt_4,172,179). coords(ricgt_4,24,141,179).  visible(ricgt_4,179). happensAt(walking(ricgt_4),179). orientation(ricgt_5,0,179). coords(ricgt_5,25,139,179).  visible(ricgt_5,179). happensAt(walking(ricgt_5),179).
orientation(ricgt_4,172,180). coords(ricgt_4,24,141,180).  visible(ricgt_4,180). happensAt(walking(ricgt_4),180). orientation(ricgt_5,0,180). coords(ricgt_5,25,139,180).  visible(ricgt_5,180). happensAt(walking(ricgt_5),180).
orientation(ricgt_4,172,181). coords(ricgt_4,24,141,181).  visible(ricgt_4,181). happensAt(walking(ricgt_4),181). orientation(ricgt_5,0,181). coords(ricgt_5,25,139,181).  happensAt(disappear(ricgt_5),181). happensAt(walking(ricgt_5),181).
#show holdsAt/2.