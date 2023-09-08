
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

	orientation(mc1gt_1,145,156). coords(mc1gt_1,128,83,156).  visible(mc1gt_1,156). happensAt(walking(mc1gt_1),156). orientation(mc1gt_3,146,156). coords(mc1gt_3,99,80,156).  visible(mc1gt_3,156). happensAt(walking(mc1gt_3),156).
orientation(mc1gt_1,145,157). coords(mc1gt_1,129,83,157).  visible(mc1gt_1,157). happensAt(walking(mc1gt_1),157). orientation(mc1gt_3,146,157). coords(mc1gt_3,99,81,157).  visible(mc1gt_3,157). happensAt(walking(mc1gt_3),157).
orientation(mc1gt_1,145,158). coords(mc1gt_1,130,84,158).  visible(mc1gt_1,158). happensAt(walking(mc1gt_1),158). orientation(mc1gt_3,146,158). coords(mc1gt_3,99,81,158).  visible(mc1gt_3,158). happensAt(walking(mc1gt_3),158).
orientation(mc1gt_1,145,159). coords(mc1gt_1,130,84,159).  visible(mc1gt_1,159). happensAt(walking(mc1gt_1),159). orientation(mc1gt_3,146,159). coords(mc1gt_3,100,81,159).  visible(mc1gt_3,159). happensAt(walking(mc1gt_3),159).
orientation(mc1gt_1,145,160). coords(mc1gt_1,130,84,160).  visible(mc1gt_1,160). happensAt(walking(mc1gt_1),160). orientation(mc1gt_3,146,160). coords(mc1gt_3,101,81,160).  visible(mc1gt_3,160). happensAt(walking(mc1gt_3),160).
orientation(mc1gt_1,145,161). coords(mc1gt_1,131,84,161).  visible(mc1gt_1,161). happensAt(walking(mc1gt_1),161). orientation(mc1gt_3,146,161). coords(mc1gt_3,102,81,161).  visible(mc1gt_3,161). happensAt(walking(mc1gt_3),161).
orientation(mc1gt_1,145,162). coords(mc1gt_1,131,84,162).  visible(mc1gt_1,162). happensAt(walking(mc1gt_1),162). orientation(mc1gt_3,146,162). coords(mc1gt_3,103,81,162).  visible(mc1gt_3,162). happensAt(walking(mc1gt_3),162).
orientation(mc1gt_1,145,163). coords(mc1gt_1,131,84,163).  visible(mc1gt_1,163). happensAt(walking(mc1gt_1),163). orientation(mc1gt_3,146,163). coords(mc1gt_3,104,81,163).  visible(mc1gt_3,163). happensAt(walking(mc1gt_3),163).
orientation(mc1gt_1,145,164). coords(mc1gt_1,131,84,164).  visible(mc1gt_1,164). happensAt(walking(mc1gt_1),164). orientation(mc1gt_3,146,164). coords(mc1gt_3,104,81,164).  visible(mc1gt_3,164). happensAt(walking(mc1gt_3),164).
orientation(mc1gt_1,145,165). coords(mc1gt_1,131,85,165).  visible(mc1gt_1,165). happensAt(walking(mc1gt_1),165). orientation(mc1gt_3,146,165). coords(mc1gt_3,105,81,165).  visible(mc1gt_3,165). happensAt(walking(mc1gt_3),165).
orientation(mc1gt_1,145,166). coords(mc1gt_1,131,86,166).  visible(mc1gt_1,166). happensAt(walking(mc1gt_1),166). orientation(mc1gt_3,146,166). coords(mc1gt_3,105,81,166).  visible(mc1gt_3,166). happensAt(walking(mc1gt_3),166).
orientation(mc1gt_1,145,167). coords(mc1gt_1,132,86,167).  visible(mc1gt_1,167). happensAt(walking(mc1gt_1),167). orientation(mc1gt_3,146,167). coords(mc1gt_3,105,81,167).  visible(mc1gt_3,167). happensAt(walking(mc1gt_3),167).
orientation(mc1gt_1,145,168). coords(mc1gt_1,133,87,168).  visible(mc1gt_1,168). happensAt(walking(mc1gt_1),168). orientation(mc1gt_3,146,168). coords(mc1gt_3,105,81,168).  visible(mc1gt_3,168). happensAt(walking(mc1gt_3),168).
orientation(mc1gt_1,145,169). coords(mc1gt_1,134,87,169).  visible(mc1gt_1,169). happensAt(walking(mc1gt_1),169). orientation(mc1gt_3,146,169). coords(mc1gt_3,105,81,169).  visible(mc1gt_3,169). happensAt(walking(mc1gt_3),169).
orientation(mc1gt_1,145,170). coords(mc1gt_1,136,88,170).  visible(mc1gt_1,170). happensAt(walking(mc1gt_1),170). orientation(mc1gt_3,146,170). coords(mc1gt_3,105,81,170).  visible(mc1gt_3,170). happensAt(walking(mc1gt_3),170).
orientation(mc1gt_1,145,171). coords(mc1gt_1,137,89,171).  visible(mc1gt_1,171). happensAt(walking(mc1gt_1),171). orientation(mc1gt_3,146,171). coords(mc1gt_3,106,81,171).  visible(mc1gt_3,171). happensAt(walking(mc1gt_3),171).
orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172). happensAt(walking(mc1gt_1),172). orientation(mc1gt_3,146,172). coords(mc1gt_3,107,81,172).  visible(mc1gt_3,172). happensAt(walking(mc1gt_3),172).
orientation(mc1gt_1,145,173). coords(mc1gt_1,138,90,173).  visible(mc1gt_1,173). happensAt(walking(mc1gt_1),173). orientation(mc1gt_3,146,173). coords(mc1gt_3,108,81,173).  visible(mc1gt_3,173). happensAt(walking(mc1gt_3),173).
orientation(mc1gt_1,145,174). coords(mc1gt_1,138,90,174).  visible(mc1gt_1,174). happensAt(walking(mc1gt_1),174). orientation(mc1gt_3,146,174). coords(mc1gt_3,109,81,174).  visible(mc1gt_3,174). happensAt(walking(mc1gt_3),174).
orientation(mc1gt_1,145,175). coords(mc1gt_1,139,90,175).  visible(mc1gt_1,175). happensAt(walking(mc1gt_1),175). orientation(mc1gt_3,146,175). coords(mc1gt_3,110,81,175).  visible(mc1gt_3,175). happensAt(walking(mc1gt_3),175).
orientation(mc1gt_1,145,176). coords(mc1gt_1,139,90,176).  visible(mc1gt_1,176). happensAt(walking(mc1gt_1),176). orientation(mc1gt_3,146,176). coords(mc1gt_3,112,81,176).  visible(mc1gt_3,176). happensAt(walking(mc1gt_3),176).
orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177). happensAt(walking(mc1gt_1),177). orientation(mc1gt_3,146,177). coords(mc1gt_3,112,81,177).  visible(mc1gt_3,177). happensAt(walking(mc1gt_3),177).
orientation(mc1gt_1,145,178). coords(mc1gt_1,139,91,178).  visible(mc1gt_1,178). happensAt(walking(mc1gt_1),178). orientation(mc1gt_3,146,178). coords(mc1gt_3,112,81,178).  visible(mc1gt_3,178). happensAt(walking(mc1gt_3),178).
orientation(mc1gt_1,145,179). coords(mc1gt_1,140,91,179).  visible(mc1gt_1,179). happensAt(walking(mc1gt_1),179). orientation(mc1gt_3,146,179). coords(mc1gt_3,113,81,179).  visible(mc1gt_3,179). happensAt(walking(mc1gt_3),179).
orientation(mc1gt_1,145,180). coords(mc1gt_1,140,91,180).  visible(mc1gt_1,180). happensAt(walking(mc1gt_1),180). orientation(mc1gt_3,146,180). coords(mc1gt_3,113,81,180).  visible(mc1gt_3,180). happensAt(walking(mc1gt_3),180).
#show holdsAt/2.