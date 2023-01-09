
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

	orientation(fomdgt3_1,150,150). coords(fomdgt3_1,107,108,150).  visible(fomdgt3_1,150). happensAt(walking(fomdgt3_1),150). orientation(fomdgt3_6,120,150). coords(fomdgt3_6,252,18,150).  happensAt(appear(fomdgt3_6),150). happensAt(walking(fomdgt3_6),150).
orientation(fomdgt3_1,150,151). coords(fomdgt3_1,107,108,151).  visible(fomdgt3_1,151). happensAt(walking(fomdgt3_1),151). orientation(fomdgt3_6,120,151). coords(fomdgt3_6,252,19,151).  visible(fomdgt3_6,151). happensAt(walking(fomdgt3_6),151).
orientation(fomdgt3_1,150,152). coords(fomdgt3_1,106,108,152).  visible(fomdgt3_1,152). happensAt(walking(fomdgt3_1),152). orientation(fomdgt3_6,120,152). coords(fomdgt3_6,252,19,152).  visible(fomdgt3_6,152). happensAt(walking(fomdgt3_6),152).
orientation(fomdgt3_1,150,153). coords(fomdgt3_1,106,108,153).  visible(fomdgt3_1,153). happensAt(walking(fomdgt3_1),153). orientation(fomdgt3_6,120,153). coords(fomdgt3_6,250,19,153).  visible(fomdgt3_6,153). happensAt(walking(fomdgt3_6),153).
orientation(fomdgt3_1,150,154). coords(fomdgt3_1,106,108,154).  visible(fomdgt3_1,154). happensAt(walking(fomdgt3_1),154). orientation(fomdgt3_6,110,154). coords(fomdgt3_6,249,21,154).  visible(fomdgt3_6,154). happensAt(walking(fomdgt3_6),154).
orientation(fomdgt3_1,150,155). coords(fomdgt3_1,106,108,155).  visible(fomdgt3_1,155). happensAt(walking(fomdgt3_1),155). orientation(fomdgt3_6,110,155). coords(fomdgt3_6,249,21,155).  visible(fomdgt3_6,155). happensAt(walking(fomdgt3_6),155).
orientation(fomdgt3_1,150,156). coords(fomdgt3_1,106,108,156).  visible(fomdgt3_1,156). happensAt(walking(fomdgt3_1),156). orientation(fomdgt3_6,110,156). coords(fomdgt3_6,249,21,156).  visible(fomdgt3_6,156). happensAt(walking(fomdgt3_6),156).
orientation(fomdgt3_1,150,157). coords(fomdgt3_1,99,108,157).  visible(fomdgt3_1,157). happensAt(walking(fomdgt3_1),157). orientation(fomdgt3_6,110,157). coords(fomdgt3_6,245,21,157).  visible(fomdgt3_6,157). happensAt(walking(fomdgt3_6),157).
orientation(fomdgt3_1,150,158). coords(fomdgt3_1,98,108,158).  visible(fomdgt3_1,158). happensAt(walking(fomdgt3_1),158). orientation(fomdgt3_6,110,158). coords(fomdgt3_6,242,21,158).  visible(fomdgt3_6,158). happensAt(walking(fomdgt3_6),158).
orientation(fomdgt3_1,150,159). coords(fomdgt3_1,98,108,159).  visible(fomdgt3_1,159). happensAt(walking(fomdgt3_1),159). orientation(fomdgt3_6,110,159). coords(fomdgt3_6,242,21,159).  visible(fomdgt3_6,159). happensAt(walking(fomdgt3_6),159).
orientation(fomdgt3_1,150,160). coords(fomdgt3_1,98,108,160).  visible(fomdgt3_1,160). happensAt(walking(fomdgt3_1),160). orientation(fomdgt3_6,110,160). coords(fomdgt3_6,242,21,160).  visible(fomdgt3_6,160). happensAt(walking(fomdgt3_6),160).
orientation(fomdgt3_1,150,161). coords(fomdgt3_1,97,108,161).  visible(fomdgt3_1,161). happensAt(walking(fomdgt3_1),161). orientation(fomdgt3_6,110,161). coords(fomdgt3_6,242,21,161).  visible(fomdgt3_6,161). happensAt(walking(fomdgt3_6),161).
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
#show holdsAt/2.