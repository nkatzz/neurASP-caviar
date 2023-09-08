
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

	orientation(lb1gt_0,136,156). coords(lb1gt_0,161,71,156).  visible(lb1gt_0,156). happensAt(walking(lb1gt_0),156). orientation(lb1gt_2,144,156). coords(lb1gt_2,103,109,156).  visible(lb1gt_2,156). happensAt(walking(lb1gt_2),156).
orientation(lb1gt_0,136,157). coords(lb1gt_0,162,69,157).  visible(lb1gt_0,157). happensAt(walking(lb1gt_0),157). orientation(lb1gt_2,144,157). coords(lb1gt_2,103,109,157).  visible(lb1gt_2,157). happensAt(walking(lb1gt_2),157).
orientation(lb1gt_0,136,158). coords(lb1gt_0,161,67,158).  visible(lb1gt_0,158). happensAt(walking(lb1gt_0),158). orientation(lb1gt_2,144,158). coords(lb1gt_2,103,110,158).  visible(lb1gt_2,158). happensAt(walking(lb1gt_2),158).
orientation(lb1gt_0,136,159). coords(lb1gt_0,162,65,159).  visible(lb1gt_0,159). happensAt(walking(lb1gt_0),159). orientation(lb1gt_2,144,159). coords(lb1gt_2,104,110,159).  visible(lb1gt_2,159). happensAt(walking(lb1gt_2),159).
orientation(lb1gt_0,136,160). coords(lb1gt_0,161,64,160).  visible(lb1gt_0,160). happensAt(walking(lb1gt_0),160). orientation(lb1gt_2,144,160). coords(lb1gt_2,104,110,160).  visible(lb1gt_2,160). happensAt(walking(lb1gt_2),160).
orientation(lb1gt_0,136,161). coords(lb1gt_0,161,63,161).  visible(lb1gt_0,161). happensAt(walking(lb1gt_0),161). orientation(lb1gt_2,144,161). coords(lb1gt_2,104,110,161).  visible(lb1gt_2,161). happensAt(walking(lb1gt_2),161).
orientation(lb1gt_0,136,162). coords(lb1gt_0,160,62,162).  visible(lb1gt_0,162). happensAt(walking(lb1gt_0),162). orientation(lb1gt_2,144,162). coords(lb1gt_2,105,110,162).  visible(lb1gt_2,162). happensAt(walking(lb1gt_2),162).
orientation(lb1gt_0,136,163). coords(lb1gt_0,160,61,163).  visible(lb1gt_0,163). happensAt(walking(lb1gt_0),163). orientation(lb1gt_2,144,163). coords(lb1gt_2,106,110,163).  visible(lb1gt_2,163). happensAt(walking(lb1gt_2),163).
orientation(lb1gt_0,136,164). coords(lb1gt_0,160,59,164).  visible(lb1gt_0,164). happensAt(walking(lb1gt_0),164). orientation(lb1gt_2,144,164). coords(lb1gt_2,108,110,164).  visible(lb1gt_2,164). happensAt(walking(lb1gt_2),164).
orientation(lb1gt_0,136,165). coords(lb1gt_0,161,59,165).  visible(lb1gt_0,165). happensAt(walking(lb1gt_0),165). orientation(lb1gt_2,144,165). coords(lb1gt_2,109,110,165).  visible(lb1gt_2,165). happensAt(walking(lb1gt_2),165).
orientation(lb1gt_0,136,166). coords(lb1gt_0,161,57,166).  visible(lb1gt_0,166). happensAt(walking(lb1gt_0),166). orientation(lb1gt_2,144,166). coords(lb1gt_2,110,111,166).  visible(lb1gt_2,166). happensAt(walking(lb1gt_2),166).
orientation(lb1gt_0,136,167). coords(lb1gt_0,161,55,167).  visible(lb1gt_0,167). happensAt(walking(lb1gt_0),167). orientation(lb1gt_2,144,167). coords(lb1gt_2,111,111,167).  visible(lb1gt_2,167). happensAt(walking(lb1gt_2),167).
orientation(lb1gt_0,136,168). coords(lb1gt_0,161,53,168).  visible(lb1gt_0,168). happensAt(walking(lb1gt_0),168). orientation(lb1gt_2,144,168). coords(lb1gt_2,111,111,168).  visible(lb1gt_2,168). happensAt(walking(lb1gt_2),168).
orientation(lb1gt_0,136,169). coords(lb1gt_0,161,51,169).  visible(lb1gt_0,169). happensAt(walking(lb1gt_0),169). orientation(lb1gt_2,144,169). coords(lb1gt_2,112,111,169).  visible(lb1gt_2,169). happensAt(walking(lb1gt_2),169).
orientation(lb1gt_0,136,170). coords(lb1gt_0,160,48,170).  visible(lb1gt_0,170). happensAt(walking(lb1gt_0),170). orientation(lb1gt_2,144,170). coords(lb1gt_2,112,111,170).  visible(lb1gt_2,170). happensAt(walking(lb1gt_2),170).
orientation(lb1gt_0,136,171). coords(lb1gt_0,160,47,171).  visible(lb1gt_0,171). happensAt(walking(lb1gt_0),171). orientation(lb1gt_2,144,171). coords(lb1gt_2,112,111,171).  visible(lb1gt_2,171). happensAt(walking(lb1gt_2),171).
orientation(lb1gt_0,136,172). coords(lb1gt_0,161,47,172).  visible(lb1gt_0,172). happensAt(walking(lb1gt_0),172). orientation(lb1gt_2,144,172). coords(lb1gt_2,113,111,172).  visible(lb1gt_2,172). happensAt(walking(lb1gt_2),172).
orientation(lb1gt_0,136,173). coords(lb1gt_0,161,47,173).  visible(lb1gt_0,173). happensAt(walking(lb1gt_0),173). orientation(lb1gt_2,144,173). coords(lb1gt_2,113,111,173).  visible(lb1gt_2,173). happensAt(walking(lb1gt_2),173).
orientation(lb1gt_0,136,174). coords(lb1gt_0,161,46,174).  visible(lb1gt_0,174). happensAt(walking(lb1gt_0),174). orientation(lb1gt_2,144,174). coords(lb1gt_2,113,111,174).  visible(lb1gt_2,174). happensAt(walking(lb1gt_2),174).
orientation(lb1gt_0,136,175). coords(lb1gt_0,161,46,175).  visible(lb1gt_0,175). happensAt(walking(lb1gt_0),175). orientation(lb1gt_2,144,175). coords(lb1gt_2,113,111,175).  visible(lb1gt_2,175). happensAt(walking(lb1gt_2),175).
orientation(lb1gt_0,136,176). coords(lb1gt_0,161,44,176).  visible(lb1gt_0,176). happensAt(walking(lb1gt_0),176). orientation(lb1gt_2,144,176). coords(lb1gt_2,113,111,176).  visible(lb1gt_2,176). happensAt(walking(lb1gt_2),176).
orientation(lb1gt_0,136,177). coords(lb1gt_0,161,43,177).  visible(lb1gt_0,177). happensAt(walking(lb1gt_0),177). orientation(lb1gt_2,144,177). coords(lb1gt_2,114,112,177).  visible(lb1gt_2,177). happensAt(walking(lb1gt_2),177).
orientation(lb1gt_0,136,178). coords(lb1gt_0,161,41,178).  visible(lb1gt_0,178). happensAt(walking(lb1gt_0),178). orientation(lb1gt_2,144,178). coords(lb1gt_2,114,112,178).  visible(lb1gt_2,178). happensAt(walking(lb1gt_2),178).
orientation(lb1gt_0,136,179). coords(lb1gt_0,161,39,179).  visible(lb1gt_0,179). happensAt(walking(lb1gt_0),179). orientation(lb1gt_2,144,179). coords(lb1gt_2,115,112,179).  visible(lb1gt_2,179). happensAt(walking(lb1gt_2),179).
orientation(lb1gt_0,136,180). coords(lb1gt_0,161,38,180).  visible(lb1gt_0,180). happensAt(walking(lb1gt_0),180). orientation(lb1gt_2,144,180). coords(lb1gt_2,115,113,180).  visible(lb1gt_2,180). happensAt(walking(lb1gt_2),180).
#show holdsAt/2.