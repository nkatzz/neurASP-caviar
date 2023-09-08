
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

	orientation(wk2gt_1,129,156). coords(wk2gt_1,114,53,156).  visible(wk2gt_1,156). happensAt(inactive(wk2gt_1),156). orientation(wk2gt_2,141,156). coords(wk2gt_2,98,62,156).  visible(wk2gt_2,156). happensAt(inactive(wk2gt_2),156).
orientation(wk2gt_1,129,157). coords(wk2gt_1,114,53,157).  visible(wk2gt_1,157). happensAt(inactive(wk2gt_1),157). orientation(wk2gt_2,141,157). coords(wk2gt_2,98,62,157).  visible(wk2gt_2,157). happensAt(inactive(wk2gt_2),157).
orientation(wk2gt_1,129,158). coords(wk2gt_1,114,53,158).  visible(wk2gt_1,158). happensAt(inactive(wk2gt_1),158). orientation(wk2gt_2,141,158). coords(wk2gt_2,98,62,158).  visible(wk2gt_2,158). happensAt(inactive(wk2gt_2),158).
orientation(wk2gt_1,129,159). coords(wk2gt_1,115,53,159).  visible(wk2gt_1,159). happensAt(inactive(wk2gt_1),159). orientation(wk2gt_2,141,159). coords(wk2gt_2,98,62,159).  visible(wk2gt_2,159). happensAt(inactive(wk2gt_2),159).
orientation(wk2gt_1,129,160). coords(wk2gt_1,115,53,160).  visible(wk2gt_1,160). happensAt(inactive(wk2gt_1),160). orientation(wk2gt_2,141,160). coords(wk2gt_2,98,62,160).  visible(wk2gt_2,160). happensAt(inactive(wk2gt_2),160).
orientation(wk2gt_1,129,161). coords(wk2gt_1,115,53,161).  visible(wk2gt_1,161). happensAt(inactive(wk2gt_1),161). orientation(wk2gt_2,141,161). coords(wk2gt_2,98,62,161).  visible(wk2gt_2,161). happensAt(inactive(wk2gt_2),161).
orientation(wk2gt_1,129,162). coords(wk2gt_1,115,53,162).  visible(wk2gt_1,162). happensAt(inactive(wk2gt_1),162). orientation(wk2gt_2,141,162). coords(wk2gt_2,98,62,162).  visible(wk2gt_2,162). happensAt(inactive(wk2gt_2),162).
orientation(wk2gt_1,129,163). coords(wk2gt_1,115,53,163).  visible(wk2gt_1,163). happensAt(inactive(wk2gt_1),163). orientation(wk2gt_2,141,163). coords(wk2gt_2,98,62,163).  visible(wk2gt_2,163). happensAt(inactive(wk2gt_2),163).
orientation(wk2gt_1,129,164). coords(wk2gt_1,115,53,164).  visible(wk2gt_1,164). happensAt(inactive(wk2gt_1),164). orientation(wk2gt_2,141,164). coords(wk2gt_2,98,62,164).  visible(wk2gt_2,164). happensAt(inactive(wk2gt_2),164).
orientation(wk2gt_1,129,165). coords(wk2gt_1,115,53,165).  visible(wk2gt_1,165). happensAt(inactive(wk2gt_1),165). orientation(wk2gt_2,141,165). coords(wk2gt_2,98,62,165).  visible(wk2gt_2,165). happensAt(inactive(wk2gt_2),165).
orientation(wk2gt_1,129,166). coords(wk2gt_1,115,53,166).  visible(wk2gt_1,166). happensAt(inactive(wk2gt_1),166). orientation(wk2gt_2,141,166). coords(wk2gt_2,98,62,166).  visible(wk2gt_2,166). happensAt(inactive(wk2gt_2),166).
orientation(wk2gt_1,129,167). coords(wk2gt_1,115,53,167).  visible(wk2gt_1,167). happensAt(inactive(wk2gt_1),167). orientation(wk2gt_2,141,167). coords(wk2gt_2,98,62,167).  visible(wk2gt_2,167). happensAt(inactive(wk2gt_2),167).
orientation(wk2gt_1,129,168). coords(wk2gt_1,115,53,168).  visible(wk2gt_1,168). happensAt(inactive(wk2gt_1),168). orientation(wk2gt_2,141,168). coords(wk2gt_2,98,62,168).  visible(wk2gt_2,168). happensAt(inactive(wk2gt_2),168).
orientation(wk2gt_1,129,169). coords(wk2gt_1,115,53,169).  visible(wk2gt_1,169). happensAt(inactive(wk2gt_1),169). orientation(wk2gt_2,141,169). coords(wk2gt_2,98,62,169).  visible(wk2gt_2,169). happensAt(inactive(wk2gt_2),169).
orientation(wk2gt_1,129,170). coords(wk2gt_1,115,53,170).  visible(wk2gt_1,170). happensAt(inactive(wk2gt_1),170). orientation(wk2gt_2,141,170). coords(wk2gt_2,98,62,170).  visible(wk2gt_2,170). happensAt(inactive(wk2gt_2),170).
orientation(wk2gt_1,129,171). coords(wk2gt_1,115,53,171).  visible(wk2gt_1,171). happensAt(inactive(wk2gt_1),171). orientation(wk2gt_2,141,171). coords(wk2gt_2,98,62,171).  visible(wk2gt_2,171). happensAt(inactive(wk2gt_2),171).
orientation(wk2gt_1,129,172). coords(wk2gt_1,115,53,172).  visible(wk2gt_1,172). happensAt(inactive(wk2gt_1),172). orientation(wk2gt_2,141,172). coords(wk2gt_2,98,62,172).  visible(wk2gt_2,172). happensAt(inactive(wk2gt_2),172).
orientation(wk2gt_1,129,173). coords(wk2gt_1,115,53,173).  visible(wk2gt_1,173). happensAt(inactive(wk2gt_1),173). orientation(wk2gt_2,141,173). coords(wk2gt_2,98,62,173).  visible(wk2gt_2,173). happensAt(inactive(wk2gt_2),173).
orientation(wk2gt_1,129,174). coords(wk2gt_1,115,53,174).  visible(wk2gt_1,174). happensAt(inactive(wk2gt_1),174). orientation(wk2gt_2,141,174). coords(wk2gt_2,98,62,174).  visible(wk2gt_2,174). happensAt(inactive(wk2gt_2),174).
orientation(wk2gt_1,129,175). coords(wk2gt_1,115,53,175).  visible(wk2gt_1,175). happensAt(inactive(wk2gt_1),175). orientation(wk2gt_2,141,175). coords(wk2gt_2,98,62,175).  visible(wk2gt_2,175). happensAt(inactive(wk2gt_2),175).
orientation(wk2gt_1,129,176). coords(wk2gt_1,115,53,176).  visible(wk2gt_1,176). happensAt(inactive(wk2gt_1),176). orientation(wk2gt_2,141,176). coords(wk2gt_2,98,62,176).  visible(wk2gt_2,176). happensAt(inactive(wk2gt_2),176).
orientation(wk2gt_1,129,177). coords(wk2gt_1,115,53,177).  visible(wk2gt_1,177). happensAt(inactive(wk2gt_1),177). orientation(wk2gt_2,141,177). coords(wk2gt_2,98,62,177).  visible(wk2gt_2,177). happensAt(inactive(wk2gt_2),177).
orientation(wk2gt_1,129,178). coords(wk2gt_1,115,53,178).  visible(wk2gt_1,178). happensAt(inactive(wk2gt_1),178). orientation(wk2gt_2,141,178). coords(wk2gt_2,98,62,178).  visible(wk2gt_2,178). happensAt(inactive(wk2gt_2),178).
orientation(wk2gt_1,129,179). coords(wk2gt_1,115,53,179).  visible(wk2gt_1,179). happensAt(inactive(wk2gt_1),179). orientation(wk2gt_2,141,179). coords(wk2gt_2,98,62,179).  visible(wk2gt_2,179). happensAt(inactive(wk2gt_2),179).
orientation(wk2gt_1,129,180). coords(wk2gt_1,115,53,180).  visible(wk2gt_1,180). happensAt(inactive(wk2gt_1),180). orientation(wk2gt_2,141,180). coords(wk2gt_2,98,62,180).  visible(wk2gt_2,180). happensAt(inactive(wk2gt_2),180).
#show holdsAt/2.