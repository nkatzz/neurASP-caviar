
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

	orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). happensAt(walking(mc1gt_0),148). orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148). happensAt(walking(mc1gt_1),148).
orientation(mc1gt_0,147,149). coords(mc1gt_0,101,107,149).  visible(mc1gt_0,149). happensAt(walking(mc1gt_0),149). orientation(mc1gt_1,145,149). coords(mc1gt_1,120,83,149).  visible(mc1gt_1,149). happensAt(walking(mc1gt_1),149).
orientation(mc1gt_0,147,150). coords(mc1gt_0,101,107,150).  visible(mc1gt_0,150). happensAt(walking(mc1gt_0),150). orientation(mc1gt_1,145,150). coords(mc1gt_1,120,83,150).  visible(mc1gt_1,150). happensAt(walking(mc1gt_1),150).
orientation(mc1gt_0,147,151). coords(mc1gt_0,102,107,151).  visible(mc1gt_0,151). happensAt(walking(mc1gt_0),151). orientation(mc1gt_1,145,151). coords(mc1gt_1,121,83,151).  visible(mc1gt_1,151). happensAt(walking(mc1gt_1),151).
orientation(mc1gt_0,147,152). coords(mc1gt_0,103,107,152).  visible(mc1gt_0,152). happensAt(walking(mc1gt_0),152). orientation(mc1gt_1,145,152). coords(mc1gt_1,123,83,152).  visible(mc1gt_1,152). happensAt(walking(mc1gt_1),152).
orientation(mc1gt_0,147,153). coords(mc1gt_0,104,107,153).  visible(mc1gt_0,153). happensAt(walking(mc1gt_0),153). orientation(mc1gt_1,145,153). coords(mc1gt_1,124,83,153).  visible(mc1gt_1,153). happensAt(walking(mc1gt_1),153).
orientation(mc1gt_0,147,154). coords(mc1gt_0,104,107,154).  visible(mc1gt_0,154). happensAt(walking(mc1gt_0),154). orientation(mc1gt_1,145,154). coords(mc1gt_1,126,83,154).  visible(mc1gt_1,154). happensAt(walking(mc1gt_1),154).
orientation(mc1gt_0,147,155). coords(mc1gt_0,104,107,155).  visible(mc1gt_0,155). happensAt(walking(mc1gt_0),155). orientation(mc1gt_1,145,155). coords(mc1gt_1,127,83,155).  visible(mc1gt_1,155). happensAt(walking(mc1gt_1),155).
orientation(mc1gt_0,147,156). coords(mc1gt_0,104,107,156).  visible(mc1gt_0,156). happensAt(walking(mc1gt_0),156). orientation(mc1gt_1,145,156). coords(mc1gt_1,128,83,156).  visible(mc1gt_1,156). happensAt(walking(mc1gt_1),156).
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). happensAt(walking(mc1gt_0),157). orientation(mc1gt_1,145,157). coords(mc1gt_1,129,83,157).  visible(mc1gt_1,157). happensAt(walking(mc1gt_1),157).
orientation(mc1gt_0,147,158). coords(mc1gt_0,104,108,158).  visible(mc1gt_0,158). happensAt(walking(mc1gt_0),158). orientation(mc1gt_1,145,158). coords(mc1gt_1,130,84,158).  visible(mc1gt_1,158). happensAt(walking(mc1gt_1),158).
orientation(mc1gt_0,147,159). coords(mc1gt_0,105,108,159).  visible(mc1gt_0,159). happensAt(walking(mc1gt_0),159). orientation(mc1gt_1,145,159). coords(mc1gt_1,130,84,159).  visible(mc1gt_1,159). happensAt(walking(mc1gt_1),159).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). happensAt(walking(mc1gt_0),160). orientation(mc1gt_1,145,160). coords(mc1gt_1,130,84,160).  visible(mc1gt_1,160). happensAt(walking(mc1gt_1),160).
orientation(mc1gt_0,147,161). coords(mc1gt_0,107,109,161).  visible(mc1gt_0,161). happensAt(walking(mc1gt_0),161). orientation(mc1gt_1,145,161). coords(mc1gt_1,131,84,161).  visible(mc1gt_1,161). happensAt(walking(mc1gt_1),161).
orientation(mc1gt_0,147,162). coords(mc1gt_0,108,109,162).  visible(mc1gt_0,162). happensAt(walking(mc1gt_0),162). orientation(mc1gt_1,145,162). coords(mc1gt_1,131,84,162).  visible(mc1gt_1,162). happensAt(walking(mc1gt_1),162).
orientation(mc1gt_0,147,163). coords(mc1gt_0,109,110,163).  visible(mc1gt_0,163). happensAt(walking(mc1gt_0),163). orientation(mc1gt_1,145,163). coords(mc1gt_1,131,84,163).  visible(mc1gt_1,163). happensAt(walking(mc1gt_1),163).
orientation(mc1gt_0,147,164). coords(mc1gt_0,110,110,164).  visible(mc1gt_0,164). happensAt(walking(mc1gt_0),164). orientation(mc1gt_1,145,164). coords(mc1gt_1,131,84,164).  visible(mc1gt_1,164). happensAt(walking(mc1gt_1),164).
orientation(mc1gt_0,147,165). coords(mc1gt_0,111,110,165).  visible(mc1gt_0,165). happensAt(walking(mc1gt_0),165). orientation(mc1gt_1,145,165). coords(mc1gt_1,131,85,165).  visible(mc1gt_1,165). happensAt(walking(mc1gt_1),165).
orientation(mc1gt_0,147,166). coords(mc1gt_0,111,110,166).  visible(mc1gt_0,166). happensAt(walking(mc1gt_0),166). orientation(mc1gt_1,145,166). coords(mc1gt_1,131,86,166).  visible(mc1gt_1,166). happensAt(walking(mc1gt_1),166).
orientation(mc1gt_0,147,167). coords(mc1gt_0,111,110,167).  visible(mc1gt_0,167). happensAt(walking(mc1gt_0),167). orientation(mc1gt_1,145,167). coords(mc1gt_1,132,86,167).  visible(mc1gt_1,167). happensAt(walking(mc1gt_1),167).
orientation(mc1gt_0,147,168). coords(mc1gt_0,111,110,168).  visible(mc1gt_0,168). happensAt(walking(mc1gt_0),168). orientation(mc1gt_1,145,168). coords(mc1gt_1,133,87,168).  visible(mc1gt_1,168). happensAt(walking(mc1gt_1),168).
orientation(mc1gt_0,147,169). coords(mc1gt_0,111,110,169).  visible(mc1gt_0,169). happensAt(walking(mc1gt_0),169). orientation(mc1gt_1,145,169). coords(mc1gt_1,134,87,169).  visible(mc1gt_1,169). happensAt(walking(mc1gt_1),169).
orientation(mc1gt_0,147,170). coords(mc1gt_0,112,110,170).  visible(mc1gt_0,170). happensAt(walking(mc1gt_0),170). orientation(mc1gt_1,145,170). coords(mc1gt_1,136,88,170).  visible(mc1gt_1,170). happensAt(walking(mc1gt_1),170).
orientation(mc1gt_0,147,171). coords(mc1gt_0,112,110,171).  visible(mc1gt_0,171). happensAt(walking(mc1gt_0),171). orientation(mc1gt_1,145,171). coords(mc1gt_1,137,89,171).  visible(mc1gt_1,171). happensAt(walking(mc1gt_1),171).
orientation(mc1gt_0,147,172). coords(mc1gt_0,113,110,172).  visible(mc1gt_0,172). happensAt(walking(mc1gt_0),172). orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172). happensAt(walking(mc1gt_1),172).
#show holdsAt/2.