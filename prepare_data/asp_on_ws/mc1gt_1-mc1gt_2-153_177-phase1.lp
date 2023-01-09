
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

	orientation(mc1gt_1,145,153). coords(mc1gt_1,124,83,153).  visible(mc1gt_1,153). happensAt(walking(mc1gt_1),153). orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153). happensAt(walking(mc1gt_2),153).
orientation(mc1gt_1,145,154). coords(mc1gt_1,126,83,154).  visible(mc1gt_1,154). happensAt(walking(mc1gt_1),154). orientation(mc1gt_2,144,154). coords(mc1gt_2,80,105,154).  visible(mc1gt_2,154). happensAt(walking(mc1gt_2),154).
orientation(mc1gt_1,145,155). coords(mc1gt_1,127,83,155).  visible(mc1gt_1,155). happensAt(walking(mc1gt_1),155). orientation(mc1gt_2,144,155). coords(mc1gt_2,80,105,155).  visible(mc1gt_2,155). happensAt(walking(mc1gt_2),155).
orientation(mc1gt_1,145,156). coords(mc1gt_1,128,83,156).  visible(mc1gt_1,156). happensAt(walking(mc1gt_1),156). orientation(mc1gt_2,144,156). coords(mc1gt_2,80,105,156).  visible(mc1gt_2,156). happensAt(walking(mc1gt_2),156).
orientation(mc1gt_1,145,157). coords(mc1gt_1,129,83,157).  visible(mc1gt_1,157). happensAt(walking(mc1gt_1),157). orientation(mc1gt_2,144,157). coords(mc1gt_2,80,105,157).  visible(mc1gt_2,157). happensAt(walking(mc1gt_2),157).
orientation(mc1gt_1,145,158). coords(mc1gt_1,130,84,158).  visible(mc1gt_1,158). happensAt(walking(mc1gt_1),158). orientation(mc1gt_2,144,158). coords(mc1gt_2,80,106,158).  visible(mc1gt_2,158). happensAt(walking(mc1gt_2),158).
orientation(mc1gt_1,145,159). coords(mc1gt_1,130,84,159).  visible(mc1gt_1,159). happensAt(walking(mc1gt_1),159). orientation(mc1gt_2,144,159). coords(mc1gt_2,80,106,159).  visible(mc1gt_2,159). happensAt(walking(mc1gt_2),159).
orientation(mc1gt_1,145,160). coords(mc1gt_1,130,84,160).  visible(mc1gt_1,160). happensAt(walking(mc1gt_1),160). orientation(mc1gt_2,144,160). coords(mc1gt_2,80,106,160).  visible(mc1gt_2,160). happensAt(walking(mc1gt_2),160).
orientation(mc1gt_1,145,161). coords(mc1gt_1,131,84,161).  visible(mc1gt_1,161). happensAt(walking(mc1gt_1),161). orientation(mc1gt_2,144,161). coords(mc1gt_2,81,106,161).  visible(mc1gt_2,161). happensAt(walking(mc1gt_2),161).
orientation(mc1gt_1,145,162). coords(mc1gt_1,131,84,162).  visible(mc1gt_1,162). happensAt(walking(mc1gt_1),162). orientation(mc1gt_2,144,162). coords(mc1gt_2,81,106,162).  visible(mc1gt_2,162). happensAt(walking(mc1gt_2),162).
orientation(mc1gt_1,145,163). coords(mc1gt_1,131,84,163).  visible(mc1gt_1,163). happensAt(walking(mc1gt_1),163). orientation(mc1gt_2,144,163). coords(mc1gt_2,82,107,163).  visible(mc1gt_2,163). happensAt(walking(mc1gt_2),163).
orientation(mc1gt_1,145,164). coords(mc1gt_1,131,84,164).  visible(mc1gt_1,164). happensAt(walking(mc1gt_1),164). orientation(mc1gt_2,144,164). coords(mc1gt_2,83,107,164).  visible(mc1gt_2,164). happensAt(walking(mc1gt_2),164).
orientation(mc1gt_1,145,165). coords(mc1gt_1,131,85,165).  visible(mc1gt_1,165). happensAt(walking(mc1gt_1),165). orientation(mc1gt_2,144,165). coords(mc1gt_2,84,107,165).  visible(mc1gt_2,165). happensAt(walking(mc1gt_2),165).
orientation(mc1gt_1,145,166). coords(mc1gt_1,131,86,166).  visible(mc1gt_1,166). happensAt(walking(mc1gt_1),166). orientation(mc1gt_2,144,166). coords(mc1gt_2,85,107,166).  visible(mc1gt_2,166). happensAt(walking(mc1gt_2),166).
orientation(mc1gt_1,145,167). coords(mc1gt_1,132,86,167).  visible(mc1gt_1,167). happensAt(walking(mc1gt_1),167). orientation(mc1gt_2,144,167). coords(mc1gt_2,86,108,167).  visible(mc1gt_2,167). happensAt(walking(mc1gt_2),167).
orientation(mc1gt_1,145,168). coords(mc1gt_1,133,87,168).  visible(mc1gt_1,168). happensAt(walking(mc1gt_1),168). orientation(mc1gt_2,144,168). coords(mc1gt_2,87,108,168).  visible(mc1gt_2,168). happensAt(walking(mc1gt_2),168).
orientation(mc1gt_1,145,169). coords(mc1gt_1,134,87,169).  visible(mc1gt_1,169). happensAt(walking(mc1gt_1),169). orientation(mc1gt_2,144,169). coords(mc1gt_2,87,108,169).  visible(mc1gt_2,169). happensAt(walking(mc1gt_2),169).
orientation(mc1gt_1,145,170). coords(mc1gt_1,136,88,170).  visible(mc1gt_1,170). happensAt(walking(mc1gt_1),170). orientation(mc1gt_2,144,170). coords(mc1gt_2,87,109,170).  visible(mc1gt_2,170). happensAt(walking(mc1gt_2),170).
orientation(mc1gt_1,145,171). coords(mc1gt_1,137,89,171).  visible(mc1gt_1,171). happensAt(walking(mc1gt_1),171). orientation(mc1gt_2,144,171). coords(mc1gt_2,87,109,171).  visible(mc1gt_2,171). happensAt(walking(mc1gt_2),171).
orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172). happensAt(walking(mc1gt_1),172). orientation(mc1gt_2,144,172). coords(mc1gt_2,87,109,172).  visible(mc1gt_2,172). happensAt(walking(mc1gt_2),172).
orientation(mc1gt_1,145,173). coords(mc1gt_1,138,90,173).  visible(mc1gt_1,173). happensAt(walking(mc1gt_1),173). orientation(mc1gt_2,144,173). coords(mc1gt_2,87,109,173).  visible(mc1gt_2,173). happensAt(walking(mc1gt_2),173).
orientation(mc1gt_1,145,174). coords(mc1gt_1,138,90,174).  visible(mc1gt_1,174). happensAt(walking(mc1gt_1),174). orientation(mc1gt_2,144,174). coords(mc1gt_2,87,109,174).  visible(mc1gt_2,174). happensAt(walking(mc1gt_2),174).
orientation(mc1gt_1,145,175). coords(mc1gt_1,139,90,175).  visible(mc1gt_1,175). happensAt(walking(mc1gt_1),175). orientation(mc1gt_2,144,175). coords(mc1gt_2,88,110,175).  visible(mc1gt_2,175). happensAt(walking(mc1gt_2),175).
orientation(mc1gt_1,145,176). coords(mc1gt_1,139,90,176).  visible(mc1gt_1,176). happensAt(walking(mc1gt_1),176). orientation(mc1gt_2,144,176). coords(mc1gt_2,88,111,176).  visible(mc1gt_2,176). happensAt(walking(mc1gt_2),176).
orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177). happensAt(walking(mc1gt_1),177). orientation(mc1gt_2,144,177). coords(mc1gt_2,88,113,177).  visible(mc1gt_2,177). happensAt(walking(mc1gt_2),177).
#show holdsAt/2.