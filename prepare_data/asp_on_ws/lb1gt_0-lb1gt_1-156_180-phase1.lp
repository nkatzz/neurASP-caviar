
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

	orientation(lb1gt_0,136,156). coords(lb1gt_0,161,71,156).  visible(lb1gt_0,156). happensAt(walking(lb1gt_0),156). orientation(lb1gt_1,151,156). coords(lb1gt_1,95,139,156).  visible(lb1gt_1,156). happensAt(walking(lb1gt_1),156).
orientation(lb1gt_0,136,157). coords(lb1gt_0,162,69,157).  visible(lb1gt_0,157). happensAt(walking(lb1gt_0),157). orientation(lb1gt_1,151,157). coords(lb1gt_1,95,140,157).  visible(lb1gt_1,157). happensAt(walking(lb1gt_1),157).
orientation(lb1gt_0,136,158). coords(lb1gt_0,161,67,158).  visible(lb1gt_0,158). happensAt(walking(lb1gt_0),158). orientation(lb1gt_1,151,158). coords(lb1gt_1,95,141,158).  visible(lb1gt_1,158). happensAt(walking(lb1gt_1),158).
orientation(lb1gt_0,136,159). coords(lb1gt_0,162,65,159).  visible(lb1gt_0,159). happensAt(walking(lb1gt_0),159). orientation(lb1gt_1,151,159). coords(lb1gt_1,95,142,159).  visible(lb1gt_1,159). happensAt(walking(lb1gt_1),159).
orientation(lb1gt_0,136,160). coords(lb1gt_0,161,64,160).  visible(lb1gt_0,160). happensAt(walking(lb1gt_0),160). orientation(lb1gt_1,151,160). coords(lb1gt_1,96,144,160).  visible(lb1gt_1,160). happensAt(walking(lb1gt_1),160).
orientation(lb1gt_0,136,161). coords(lb1gt_0,161,63,161).  visible(lb1gt_0,161). happensAt(walking(lb1gt_0),161). orientation(lb1gt_1,151,161). coords(lb1gt_1,97,144,161).  visible(lb1gt_1,161). happensAt(walking(lb1gt_1),161).
orientation(lb1gt_0,136,162). coords(lb1gt_0,160,62,162).  visible(lb1gt_0,162). happensAt(walking(lb1gt_0),162). orientation(lb1gt_1,151,162). coords(lb1gt_1,98,145,162).  visible(lb1gt_1,162). happensAt(walking(lb1gt_1),162).
orientation(lb1gt_0,136,163). coords(lb1gt_0,160,61,163).  visible(lb1gt_0,163). happensAt(walking(lb1gt_0),163). orientation(lb1gt_1,151,163). coords(lb1gt_1,98,145,163).  visible(lb1gt_1,163). happensAt(walking(lb1gt_1),163).
orientation(lb1gt_0,136,164). coords(lb1gt_0,160,59,164).  visible(lb1gt_0,164). happensAt(walking(lb1gt_0),164). orientation(lb1gt_1,151,164). coords(lb1gt_1,98,145,164).  visible(lb1gt_1,164). happensAt(walking(lb1gt_1),164).
orientation(lb1gt_0,136,165). coords(lb1gt_0,161,59,165).  visible(lb1gt_0,165). happensAt(walking(lb1gt_0),165). orientation(lb1gt_1,151,165). coords(lb1gt_1,100,146,165).  visible(lb1gt_1,165). happensAt(walking(lb1gt_1),165).
orientation(lb1gt_0,136,166). coords(lb1gt_0,161,57,166).  visible(lb1gt_0,166). happensAt(walking(lb1gt_0),166). orientation(lb1gt_1,151,166). coords(lb1gt_1,101,147,166).  visible(lb1gt_1,166). happensAt(walking(lb1gt_1),166).
orientation(lb1gt_0,136,167). coords(lb1gt_0,161,55,167).  visible(lb1gt_0,167). happensAt(walking(lb1gt_0),167). orientation(lb1gt_1,151,167). coords(lb1gt_1,101,147,167).  visible(lb1gt_1,167). happensAt(walking(lb1gt_1),167).
orientation(lb1gt_0,136,168). coords(lb1gt_0,161,53,168).  visible(lb1gt_0,168). happensAt(walking(lb1gt_0),168). orientation(lb1gt_1,151,168). coords(lb1gt_1,101,147,168).  visible(lb1gt_1,168). happensAt(walking(lb1gt_1),168).
orientation(lb1gt_0,136,169). coords(lb1gt_0,161,51,169).  visible(lb1gt_0,169). happensAt(walking(lb1gt_0),169). orientation(lb1gt_1,151,169). coords(lb1gt_1,101,147,169).  visible(lb1gt_1,169). happensAt(walking(lb1gt_1),169).
orientation(lb1gt_0,136,170). coords(lb1gt_0,160,48,170).  visible(lb1gt_0,170). happensAt(walking(lb1gt_0),170). orientation(lb1gt_1,151,170). coords(lb1gt_1,101,148,170).  visible(lb1gt_1,170). happensAt(walking(lb1gt_1),170).
orientation(lb1gt_0,136,171). coords(lb1gt_0,160,47,171).  visible(lb1gt_0,171). happensAt(walking(lb1gt_0),171). orientation(lb1gt_1,151,171). coords(lb1gt_1,101,148,171).  visible(lb1gt_1,171). happensAt(walking(lb1gt_1),171).
orientation(lb1gt_0,136,172). coords(lb1gt_0,161,47,172).  visible(lb1gt_0,172). happensAt(walking(lb1gt_0),172). orientation(lb1gt_1,151,172). coords(lb1gt_1,102,148,172).  visible(lb1gt_1,172). happensAt(walking(lb1gt_1),172).
orientation(lb1gt_0,136,173). coords(lb1gt_0,161,47,173).  visible(lb1gt_0,173). happensAt(walking(lb1gt_0),173). orientation(lb1gt_1,151,173). coords(lb1gt_1,102,148,173).  visible(lb1gt_1,173). happensAt(walking(lb1gt_1),173).
orientation(lb1gt_0,136,174). coords(lb1gt_0,161,46,174).  visible(lb1gt_0,174). happensAt(walking(lb1gt_0),174). orientation(lb1gt_1,151,174). coords(lb1gt_1,102,148,174).  visible(lb1gt_1,174). happensAt(walking(lb1gt_1),174).
orientation(lb1gt_0,136,175). coords(lb1gt_0,161,46,175).  visible(lb1gt_0,175). happensAt(walking(lb1gt_0),175). orientation(lb1gt_1,151,175). coords(lb1gt_1,102,149,175).  visible(lb1gt_1,175). happensAt(walking(lb1gt_1),175).
orientation(lb1gt_0,136,176). coords(lb1gt_0,161,44,176).  visible(lb1gt_0,176). happensAt(walking(lb1gt_0),176). orientation(lb1gt_1,151,176). coords(lb1gt_1,103,149,176).  visible(lb1gt_1,176). happensAt(walking(lb1gt_1),176).
orientation(lb1gt_0,136,177). coords(lb1gt_0,161,43,177).  visible(lb1gt_0,177). happensAt(walking(lb1gt_0),177). orientation(lb1gt_1,151,177). coords(lb1gt_1,105,149,177).  visible(lb1gt_1,177). happensAt(walking(lb1gt_1),177).
orientation(lb1gt_0,136,178). coords(lb1gt_0,161,41,178).  visible(lb1gt_0,178). happensAt(walking(lb1gt_0),178). orientation(lb1gt_1,151,178). coords(lb1gt_1,106,149,178).  visible(lb1gt_1,178). happensAt(walking(lb1gt_1),178).
orientation(lb1gt_0,136,179). coords(lb1gt_0,161,39,179).  visible(lb1gt_0,179). happensAt(walking(lb1gt_0),179). orientation(lb1gt_1,151,179). coords(lb1gt_1,107,149,179).  visible(lb1gt_1,179). happensAt(walking(lb1gt_1),179).
orientation(lb1gt_0,136,180). coords(lb1gt_0,161,38,180).  visible(lb1gt_0,180). happensAt(walking(lb1gt_0),180). orientation(lb1gt_1,151,180). coords(lb1gt_1,108,149,180).  visible(lb1gt_1,180). happensAt(walking(lb1gt_1),180).
#show holdsAt/2.