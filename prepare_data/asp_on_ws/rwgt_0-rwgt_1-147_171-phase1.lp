
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

	orientation(rwgt_0,142,147). coords(rwgt_0,178,121,147).  visible(rwgt_0,147). happensAt(active(rwgt_0),147). orientation(rwgt_1,138,147). coords(rwgt_1,144,81,147).  visible(rwgt_1,147). happensAt(walking(rwgt_1),147).
orientation(rwgt_0,144,148). coords(rwgt_0,178,120,148).  visible(rwgt_0,148). happensAt(active(rwgt_0),148). orientation(rwgt_1,140,148). coords(rwgt_1,144,80,148).  visible(rwgt_1,148). happensAt(walking(rwgt_1),148).
orientation(rwgt_0,143,149). coords(rwgt_0,178,119,149).  visible(rwgt_0,149). happensAt(active(rwgt_0),149). orientation(rwgt_1,140,149). coords(rwgt_1,144,80,149).  visible(rwgt_1,149). happensAt(walking(rwgt_1),149).
orientation(rwgt_0,147,150). coords(rwgt_0,178,119,150).  visible(rwgt_0,150). happensAt(active(rwgt_0),150). orientation(rwgt_1,135,150). coords(rwgt_1,144,79,150).  visible(rwgt_1,150). happensAt(walking(rwgt_1),150).
orientation(rwgt_0,147,151). coords(rwgt_0,178,119,151).  visible(rwgt_0,151). happensAt(active(rwgt_0),151). orientation(rwgt_1,135,151). coords(rwgt_1,143,78,151).  visible(rwgt_1,151). happensAt(walking(rwgt_1),151).
orientation(rwgt_0,149,152). coords(rwgt_0,178,119,152).  visible(rwgt_0,152). happensAt(active(rwgt_0),152). orientation(rwgt_1,132,152). coords(rwgt_1,142,76,152).  visible(rwgt_1,152). happensAt(walking(rwgt_1),152).
orientation(rwgt_0,149,153). coords(rwgt_0,178,119,153).  visible(rwgt_0,153). happensAt(active(rwgt_0),153). orientation(rwgt_1,132,153). coords(rwgt_1,142,74,153).  visible(rwgt_1,153). happensAt(walking(rwgt_1),153).
orientation(rwgt_0,149,154). coords(rwgt_0,177,118,154).  visible(rwgt_0,154). happensAt(active(rwgt_0),154). orientation(rwgt_1,132,154). coords(rwgt_1,142,73,154).  visible(rwgt_1,154). happensAt(walking(rwgt_1),154).
orientation(rwgt_0,149,155). coords(rwgt_0,176,118,155).  visible(rwgt_0,155). happensAt(active(rwgt_0),155). orientation(rwgt_1,135,155). coords(rwgt_1,142,71,155).  visible(rwgt_1,155). happensAt(walking(rwgt_1),155).
orientation(rwgt_0,149,156). coords(rwgt_0,175,118,156).  visible(rwgt_0,156). happensAt(active(rwgt_0),156). orientation(rwgt_1,135,156). coords(rwgt_1,142,68,156).  visible(rwgt_1,156). happensAt(walking(rwgt_1),156).
orientation(rwgt_0,149,157). coords(rwgt_0,174,118,157).  visible(rwgt_0,157). happensAt(active(rwgt_0),157). orientation(rwgt_1,135,157). coords(rwgt_1,142,66,157).  visible(rwgt_1,157). happensAt(walking(rwgt_1),157).
orientation(rwgt_0,149,158). coords(rwgt_0,173,117,158).  visible(rwgt_0,158). happensAt(active(rwgt_0),158). orientation(rwgt_1,135,158). coords(rwgt_1,142,66,158).  visible(rwgt_1,158). happensAt(walking(rwgt_1),158).
orientation(rwgt_0,149,159). coords(rwgt_0,171,117,159).  visible(rwgt_0,159). happensAt(active(rwgt_0),159). orientation(rwgt_1,135,159). coords(rwgt_1,141,66,159).  visible(rwgt_1,159). happensAt(walking(rwgt_1),159).
orientation(rwgt_0,149,160). coords(rwgt_0,170,117,160).  visible(rwgt_0,160). happensAt(inactive(rwgt_0),160). orientation(rwgt_1,135,160). coords(rwgt_1,141,65,160).  visible(rwgt_1,160). happensAt(walking(rwgt_1),160).
orientation(rwgt_0,149,161). coords(rwgt_0,170,117,161).  visible(rwgt_0,161). happensAt(inactive(rwgt_0),161). orientation(rwgt_1,135,161). coords(rwgt_1,141,64,161).  visible(rwgt_1,161). happensAt(walking(rwgt_1),161).
orientation(rwgt_0,149,162). coords(rwgt_0,170,117,162).  visible(rwgt_0,162). happensAt(inactive(rwgt_0),162). orientation(rwgt_1,135,162). coords(rwgt_1,141,64,162).  visible(rwgt_1,162). happensAt(walking(rwgt_1),162).
orientation(rwgt_0,149,163). coords(rwgt_0,170,117,163).  visible(rwgt_0,163). happensAt(inactive(rwgt_0),163). orientation(rwgt_1,135,163). coords(rwgt_1,141,63,163).  visible(rwgt_1,163). happensAt(walking(rwgt_1),163).
orientation(rwgt_0,149,164). coords(rwgt_0,170,117,164).  visible(rwgt_0,164). happensAt(inactive(rwgt_0),164). orientation(rwgt_1,135,164). coords(rwgt_1,140,62,164).  visible(rwgt_1,164). happensAt(walking(rwgt_1),164).
orientation(rwgt_0,149,165). coords(rwgt_0,170,117,165).  visible(rwgt_0,165). happensAt(inactive(rwgt_0),165). orientation(rwgt_1,135,165). coords(rwgt_1,140,61,165).  visible(rwgt_1,165). happensAt(walking(rwgt_1),165).
orientation(rwgt_0,149,166). coords(rwgt_0,170,117,166).  visible(rwgt_0,166). happensAt(inactive(rwgt_0),166). orientation(rwgt_1,135,166). coords(rwgt_1,140,59,166).  visible(rwgt_1,166). happensAt(walking(rwgt_1),166).
orientation(rwgt_0,149,167). coords(rwgt_0,170,117,167).  visible(rwgt_0,167). happensAt(inactive(rwgt_0),167). orientation(rwgt_1,135,167). coords(rwgt_1,139,58,167).  visible(rwgt_1,167). happensAt(walking(rwgt_1),167).
orientation(rwgt_0,149,168). coords(rwgt_0,170,117,168).  visible(rwgt_0,168). happensAt(inactive(rwgt_0),168). orientation(rwgt_1,135,168). coords(rwgt_1,138,57,168).  visible(rwgt_1,168). happensAt(walking(rwgt_1),168).
orientation(rwgt_0,149,169). coords(rwgt_0,170,117,169).  visible(rwgt_0,169). happensAt(inactive(rwgt_0),169). orientation(rwgt_1,135,169). coords(rwgt_1,138,56,169).  visible(rwgt_1,169). happensAt(walking(rwgt_1),169).
orientation(rwgt_0,149,170). coords(rwgt_0,170,117,170).  visible(rwgt_0,170). happensAt(inactive(rwgt_0),170). orientation(rwgt_1,135,170). coords(rwgt_1,138,56,170).  visible(rwgt_1,170). happensAt(walking(rwgt_1),170).
orientation(rwgt_0,149,171). coords(rwgt_0,170,117,171).  visible(rwgt_0,171). happensAt(inactive(rwgt_0),171). orientation(rwgt_1,135,171). coords(rwgt_1,138,56,171).  visible(rwgt_1,171). happensAt(walking(rwgt_1),171).
#show holdsAt/2.