
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

	orientation(fcgt_2,160,150). coords(fcgt_2,25,209,150).  visible(fcgt_2,150). happensAt(walking(fcgt_2),150). orientation(fcgt_4,170,150). coords(fcgt_4,101,210,150).  visible(fcgt_4,150). happensAt(walking(fcgt_4),150).
orientation(fcgt_2,160,151). coords(fcgt_2,25,209,151).  visible(fcgt_2,151). happensAt(walking(fcgt_2),151). orientation(fcgt_4,170,151). coords(fcgt_4,101,210,151).  visible(fcgt_4,151). happensAt(walking(fcgt_4),151).
orientation(fcgt_2,160,152). coords(fcgt_2,25,208,152).  visible(fcgt_2,152). happensAt(walking(fcgt_2),152). orientation(fcgt_4,170,152). coords(fcgt_4,101,210,152).  visible(fcgt_4,152). happensAt(walking(fcgt_4),152).
orientation(fcgt_2,160,153). coords(fcgt_2,26,208,153).  visible(fcgt_2,153). happensAt(walking(fcgt_2),153). orientation(fcgt_4,170,153). coords(fcgt_4,101,210,153).  visible(fcgt_4,153). happensAt(walking(fcgt_4),153).
orientation(fcgt_2,160,154). coords(fcgt_2,28,207,154).  visible(fcgt_2,154). happensAt(walking(fcgt_2),154). orientation(fcgt_4,170,154). coords(fcgt_4,101,210,154).  visible(fcgt_4,154). happensAt(walking(fcgt_4),154).
orientation(fcgt_2,160,155). coords(fcgt_2,28,208,155).  visible(fcgt_2,155). happensAt(walking(fcgt_2),155). orientation(fcgt_4,170,155). coords(fcgt_4,100,210,155).  visible(fcgt_4,155). happensAt(walking(fcgt_4),155).
orientation(fcgt_2,160,156). coords(fcgt_2,28,208,156).  visible(fcgt_2,156). happensAt(walking(fcgt_2),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
orientation(fcgt_2,160,157). coords(fcgt_2,28,208,157).  visible(fcgt_2,157). happensAt(walking(fcgt_2),157). orientation(fcgt_4,170,157). coords(fcgt_4,100,210,157).  visible(fcgt_4,157). happensAt(walking(fcgt_4),157).
orientation(fcgt_2,160,158). coords(fcgt_2,28,208,158).  visible(fcgt_2,158). happensAt(walking(fcgt_2),158). orientation(fcgt_4,170,158). coords(fcgt_4,100,210,158).  visible(fcgt_4,158). happensAt(walking(fcgt_4),158).
orientation(fcgt_2,160,159). coords(fcgt_2,28,208,159).  visible(fcgt_2,159). happensAt(walking(fcgt_2),159). orientation(fcgt_4,170,159). coords(fcgt_4,100,210,159).  visible(fcgt_4,159). happensAt(walking(fcgt_4),159).
orientation(fcgt_2,160,160). coords(fcgt_2,28,208,160).  visible(fcgt_2,160). happensAt(walking(fcgt_2),160). orientation(fcgt_4,170,160). coords(fcgt_4,100,210,160).  visible(fcgt_4,160). happensAt(walking(fcgt_4),160).
orientation(fcgt_2,160,161). coords(fcgt_2,28,207,161).  visible(fcgt_2,161). happensAt(walking(fcgt_2),161). orientation(fcgt_4,170,161). coords(fcgt_4,100,210,161).  visible(fcgt_4,161). happensAt(walking(fcgt_4),161).
orientation(fcgt_2,160,162). coords(fcgt_2,28,207,162).  visible(fcgt_2,162). happensAt(walking(fcgt_2),162). orientation(fcgt_4,165,162). coords(fcgt_4,99,210,162).  visible(fcgt_4,162). happensAt(walking(fcgt_4),162).
orientation(fcgt_2,166,163). coords(fcgt_2,28,207,163).  visible(fcgt_2,163). happensAt(walking(fcgt_2),163). orientation(fcgt_4,165,163). coords(fcgt_4,99,210,163).  visible(fcgt_4,163). happensAt(walking(fcgt_4),163).
orientation(fcgt_2,166,164). coords(fcgt_2,29,207,164).  visible(fcgt_2,164). happensAt(walking(fcgt_2),164). orientation(fcgt_4,165,164). coords(fcgt_4,99,210,164).  visible(fcgt_4,164). happensAt(walking(fcgt_4),164).
orientation(fcgt_2,166,165). coords(fcgt_2,29,207,165).  visible(fcgt_2,165). happensAt(walking(fcgt_2),165). orientation(fcgt_4,164,165). coords(fcgt_4,99,209,165).  visible(fcgt_4,165). happensAt(walking(fcgt_4),165).
orientation(fcgt_2,166,166). coords(fcgt_2,30,207,166).  visible(fcgt_2,166). happensAt(walking(fcgt_2),166). orientation(fcgt_4,165,166). coords(fcgt_4,99,209,166).  visible(fcgt_4,166). happensAt(walking(fcgt_4),166).
orientation(fcgt_2,166,167). coords(fcgt_2,30,207,167).  visible(fcgt_2,167). happensAt(walking(fcgt_2),167). orientation(fcgt_4,165,167). coords(fcgt_4,99,209,167).  visible(fcgt_4,167). happensAt(walking(fcgt_4),167).
orientation(fcgt_2,166,168). coords(fcgt_2,30,207,168).  visible(fcgt_2,168). happensAt(walking(fcgt_2),168). orientation(fcgt_4,163,168). coords(fcgt_4,99,209,168).  visible(fcgt_4,168). happensAt(walking(fcgt_4),168).
orientation(fcgt_2,166,169). coords(fcgt_2,30,207,169).  visible(fcgt_2,169). happensAt(walking(fcgt_2),169). orientation(fcgt_4,163,169). coords(fcgt_4,98,208,169).  visible(fcgt_4,169). happensAt(walking(fcgt_4),169).
orientation(fcgt_2,166,170). coords(fcgt_2,31,206,170).  visible(fcgt_2,170). happensAt(walking(fcgt_2),170). orientation(fcgt_4,165,170). coords(fcgt_4,98,208,170).  visible(fcgt_4,170). happensAt(walking(fcgt_4),170).
orientation(fcgt_2,166,171). coords(fcgt_2,32,206,171).  visible(fcgt_2,171). happensAt(walking(fcgt_2),171). orientation(fcgt_4,162,171). coords(fcgt_4,98,208,171).  visible(fcgt_4,171). happensAt(walking(fcgt_4),171).
orientation(fcgt_2,166,172). coords(fcgt_2,33,206,172).  visible(fcgt_2,172). happensAt(walking(fcgt_2),172). orientation(fcgt_4,163,172). coords(fcgt_4,97,208,172).  visible(fcgt_4,172). happensAt(walking(fcgt_4),172).
orientation(fcgt_2,166,173). coords(fcgt_2,33,205,173).  visible(fcgt_2,173). happensAt(walking(fcgt_2),173). orientation(fcgt_4,163,173). coords(fcgt_4,96,208,173).  visible(fcgt_4,173). happensAt(walking(fcgt_4),173).
orientation(fcgt_2,166,174). coords(fcgt_2,33,204,174).  visible(fcgt_2,174). happensAt(walking(fcgt_2),174). orientation(fcgt_4,163,174). coords(fcgt_4,96,207,174).  visible(fcgt_4,174). happensAt(walking(fcgt_4),174).
#show holdsAt/2.