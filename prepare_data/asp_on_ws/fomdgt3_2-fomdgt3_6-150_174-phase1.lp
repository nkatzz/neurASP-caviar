
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

	orientation(fomdgt3_2,10,150). coords(fomdgt3_2,262,228,150).  visible(fomdgt3_2,150). happensAt(walking(fomdgt3_2),150). orientation(fomdgt3_6,120,150). coords(fomdgt3_6,252,18,150).  happensAt(appear(fomdgt3_6),150). happensAt(walking(fomdgt3_6),150).
orientation(fomdgt3_2,10,151). coords(fomdgt3_2,260,226,151).  visible(fomdgt3_2,151). happensAt(walking(fomdgt3_2),151). orientation(fomdgt3_6,120,151). coords(fomdgt3_6,252,19,151).  visible(fomdgt3_6,151). happensAt(walking(fomdgt3_6),151).
orientation(fomdgt3_2,18,152). coords(fomdgt3_2,260,223,152).  visible(fomdgt3_2,152). happensAt(walking(fomdgt3_2),152). orientation(fomdgt3_6,120,152). coords(fomdgt3_6,252,19,152).  visible(fomdgt3_6,152). happensAt(walking(fomdgt3_6),152).
orientation(fomdgt3_2,18,153). coords(fomdgt3_2,260,221,153).  visible(fomdgt3_2,153). happensAt(walking(fomdgt3_2),153). orientation(fomdgt3_6,120,153). coords(fomdgt3_6,250,19,153).  visible(fomdgt3_6,153). happensAt(walking(fomdgt3_6),153).
orientation(fomdgt3_2,18,154). coords(fomdgt3_2,260,221,154).  visible(fomdgt3_2,154). happensAt(walking(fomdgt3_2),154). orientation(fomdgt3_6,110,154). coords(fomdgt3_6,249,21,154).  visible(fomdgt3_6,154). happensAt(walking(fomdgt3_6),154).
orientation(fomdgt3_2,18,155). coords(fomdgt3_2,260,221,155).  visible(fomdgt3_2,155). happensAt(walking(fomdgt3_2),155). orientation(fomdgt3_6,110,155). coords(fomdgt3_6,249,21,155).  visible(fomdgt3_6,155). happensAt(walking(fomdgt3_6),155).
orientation(fomdgt3_2,18,156). coords(fomdgt3_2,260,221,156).  visible(fomdgt3_2,156). happensAt(walking(fomdgt3_2),156). orientation(fomdgt3_6,110,156). coords(fomdgt3_6,249,21,156).  visible(fomdgt3_6,156). happensAt(walking(fomdgt3_6),156).
orientation(fomdgt3_2,18,157). coords(fomdgt3_2,260,221,157).  visible(fomdgt3_2,157). happensAt(walking(fomdgt3_2),157). orientation(fomdgt3_6,110,157). coords(fomdgt3_6,245,21,157).  visible(fomdgt3_6,157). happensAt(walking(fomdgt3_6),157).
orientation(fomdgt3_2,170,158). coords(fomdgt3_2,260,218,158).  visible(fomdgt3_2,158). happensAt(walking(fomdgt3_2),158). orientation(fomdgt3_6,110,158). coords(fomdgt3_6,242,21,158).  visible(fomdgt3_6,158). happensAt(walking(fomdgt3_6),158).
orientation(fomdgt3_2,160,159). coords(fomdgt3_2,260,215,159).  visible(fomdgt3_2,159). happensAt(walking(fomdgt3_2),159). orientation(fomdgt3_6,110,159). coords(fomdgt3_6,242,21,159).  visible(fomdgt3_6,159). happensAt(walking(fomdgt3_6),159).
orientation(fomdgt3_2,160,160). coords(fomdgt3_2,257,214,160).  visible(fomdgt3_2,160). happensAt(walking(fomdgt3_2),160). orientation(fomdgt3_6,110,160). coords(fomdgt3_6,242,21,160).  visible(fomdgt3_6,160). happensAt(walking(fomdgt3_6),160).
orientation(fomdgt3_2,160,161). coords(fomdgt3_2,257,210,161).  visible(fomdgt3_2,161). happensAt(walking(fomdgt3_2),161). orientation(fomdgt3_6,110,161). coords(fomdgt3_6,242,21,161).  visible(fomdgt3_6,161). happensAt(walking(fomdgt3_6),161).
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
#show holdsAt/2.