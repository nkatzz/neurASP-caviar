
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

	orientation(rsfgt_0,160,144). coords(rsfgt_0,35,155,144).  visible(rsfgt_0,144). happensAt(walking(rsfgt_0),144). orientation(rsfgt_1,150,144). coords(rsfgt_1,171,113,144).  visible(rsfgt_1,144). happensAt(walking(rsfgt_1),144).
orientation(rsfgt_0,160,145). coords(rsfgt_0,35,155,145).  visible(rsfgt_0,145). happensAt(walking(rsfgt_0),145). orientation(rsfgt_1,150,145). coords(rsfgt_1,171,116,145).  visible(rsfgt_1,145). happensAt(walking(rsfgt_1),145).
orientation(rsfgt_0,160,146). coords(rsfgt_0,35,155,146).  visible(rsfgt_0,146). happensAt(walking(rsfgt_0),146). orientation(rsfgt_1,150,146). coords(rsfgt_1,175,117,146).  visible(rsfgt_1,146). happensAt(walking(rsfgt_1),146).
orientation(rsfgt_0,160,147). coords(rsfgt_0,35,155,147).  visible(rsfgt_0,147). happensAt(walking(rsfgt_0),147). orientation(rsfgt_1,150,147). coords(rsfgt_1,176,119,147).  visible(rsfgt_1,147). happensAt(walking(rsfgt_1),147).
orientation(rsfgt_0,160,148). coords(rsfgt_0,35,155,148).  visible(rsfgt_0,148). happensAt(walking(rsfgt_0),148). orientation(rsfgt_1,150,148). coords(rsfgt_1,176,119,148).  visible(rsfgt_1,148). happensAt(walking(rsfgt_1),148).
orientation(rsfgt_0,160,149). coords(rsfgt_0,35,155,149).  visible(rsfgt_0,149). happensAt(walking(rsfgt_0),149). orientation(rsfgt_1,150,149). coords(rsfgt_1,176,119,149).  visible(rsfgt_1,149). happensAt(walking(rsfgt_1),149).
orientation(rsfgt_0,160,150). coords(rsfgt_0,35,155,150).  visible(rsfgt_0,150). happensAt(walking(rsfgt_0),150). orientation(rsfgt_1,150,150). coords(rsfgt_1,176,124,150).  visible(rsfgt_1,150). happensAt(walking(rsfgt_1),150).
orientation(rsfgt_0,160,151). coords(rsfgt_0,35,155,151).  visible(rsfgt_0,151). happensAt(walking(rsfgt_0),151). orientation(rsfgt_1,150,151). coords(rsfgt_1,176,128,151).  visible(rsfgt_1,151). happensAt(walking(rsfgt_1),151).
orientation(rsfgt_0,160,152). coords(rsfgt_0,35,155,152).  visible(rsfgt_0,152). happensAt(walking(rsfgt_0),152). orientation(rsfgt_1,150,152). coords(rsfgt_1,176,129,152).  visible(rsfgt_1,152). happensAt(walking(rsfgt_1),152).
orientation(rsfgt_0,160,153). coords(rsfgt_0,35,155,153).  visible(rsfgt_0,153). happensAt(walking(rsfgt_0),153). orientation(rsfgt_1,150,153). coords(rsfgt_1,176,129,153).  visible(rsfgt_1,153). happensAt(walking(rsfgt_1),153).
orientation(rsfgt_0,160,154). coords(rsfgt_0,34,155,154).  visible(rsfgt_0,154). happensAt(walking(rsfgt_0),154). orientation(rsfgt_1,150,154). coords(rsfgt_1,179,130,154).  visible(rsfgt_1,154). happensAt(walking(rsfgt_1),154).
orientation(rsfgt_0,160,155). coords(rsfgt_0,34,155,155).  visible(rsfgt_0,155). happensAt(walking(rsfgt_0),155). orientation(rsfgt_1,150,155). coords(rsfgt_1,179,131,155).  visible(rsfgt_1,155). happensAt(walking(rsfgt_1),155).
orientation(rsfgt_0,160,156). coords(rsfgt_0,34,155,156).  visible(rsfgt_0,156). happensAt(walking(rsfgt_0),156). orientation(rsfgt_1,150,156). coords(rsfgt_1,179,131,156).  visible(rsfgt_1,156). happensAt(walking(rsfgt_1),156).
orientation(rsfgt_0,160,157). coords(rsfgt_0,34,155,157).  visible(rsfgt_0,157). happensAt(walking(rsfgt_0),157). orientation(rsfgt_1,150,157). coords(rsfgt_1,179,131,157).  visible(rsfgt_1,157). happensAt(walking(rsfgt_1),157).
orientation(rsfgt_0,160,158). coords(rsfgt_0,34,155,158).  visible(rsfgt_0,158). happensAt(walking(rsfgt_0),158). orientation(rsfgt_1,150,158). coords(rsfgt_1,181,133,158).  visible(rsfgt_1,158). happensAt(walking(rsfgt_1),158).
orientation(rsfgt_0,160,159). coords(rsfgt_0,34,157,159).  visible(rsfgt_0,159). happensAt(walking(rsfgt_0),159). orientation(rsfgt_1,150,159). coords(rsfgt_1,181,133,159).  visible(rsfgt_1,159). happensAt(walking(rsfgt_1),159).
orientation(rsfgt_0,160,160). coords(rsfgt_0,34,157,160).  visible(rsfgt_0,160). happensAt(walking(rsfgt_0),160). orientation(rsfgt_1,150,160). coords(rsfgt_1,183,135,160).  visible(rsfgt_1,160). happensAt(walking(rsfgt_1),160).
orientation(rsfgt_0,160,161). coords(rsfgt_0,32,157,161).  visible(rsfgt_0,161). happensAt(walking(rsfgt_0),161). orientation(rsfgt_1,150,161). coords(rsfgt_1,186,137,161).  visible(rsfgt_1,161). happensAt(walking(rsfgt_1),161).
orientation(rsfgt_0,160,162). coords(rsfgt_0,32,157,162).  visible(rsfgt_0,162). happensAt(walking(rsfgt_0),162). orientation(rsfgt_1,150,162). coords(rsfgt_1,189,138,162).  visible(rsfgt_1,162). happensAt(walking(rsfgt_1),162).
orientation(rsfgt_0,160,163). coords(rsfgt_0,32,157,163).  visible(rsfgt_0,163). happensAt(walking(rsfgt_0),163). orientation(rsfgt_1,150,163). coords(rsfgt_1,192,141,163).  visible(rsfgt_1,163). happensAt(walking(rsfgt_1),163).
orientation(rsfgt_0,160,164). coords(rsfgt_0,32,157,164).  visible(rsfgt_0,164). happensAt(walking(rsfgt_0),164). orientation(rsfgt_1,150,164). coords(rsfgt_1,194,143,164).  visible(rsfgt_1,164). happensAt(walking(rsfgt_1),164).
orientation(rsfgt_0,160,165). coords(rsfgt_0,32,157,165).  visible(rsfgt_0,165). happensAt(walking(rsfgt_0),165). orientation(rsfgt_1,150,165). coords(rsfgt_1,195,144,165).  visible(rsfgt_1,165). happensAt(walking(rsfgt_1),165).
orientation(rsfgt_0,160,166). coords(rsfgt_0,31,157,166).  visible(rsfgt_0,166). happensAt(walking(rsfgt_0),166). orientation(rsfgt_1,150,166). coords(rsfgt_1,196,145,166).  visible(rsfgt_1,166). happensAt(walking(rsfgt_1),166).
orientation(rsfgt_0,160,167). coords(rsfgt_0,30,157,167).  visible(rsfgt_0,167). happensAt(walking(rsfgt_0),167). orientation(rsfgt_1,150,167). coords(rsfgt_1,198,147,167).  visible(rsfgt_1,167). happensAt(walking(rsfgt_1),167).
orientation(rsfgt_0,160,168). coords(rsfgt_0,30,159,168).  visible(rsfgt_0,168). happensAt(walking(rsfgt_0),168). orientation(rsfgt_1,150,168). coords(rsfgt_1,199,148,168).  visible(rsfgt_1,168). happensAt(walking(rsfgt_1),168).
#show holdsAt/2.