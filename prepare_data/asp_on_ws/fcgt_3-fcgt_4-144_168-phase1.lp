
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

	orientation(fcgt_3,159,144). coords(fcgt_3,103,163,144).  visible(fcgt_3,144). happensAt(inactive(fcgt_3),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
orientation(fcgt_3,159,145). coords(fcgt_3,103,163,145).  visible(fcgt_3,145). happensAt(inactive(fcgt_3),145). orientation(fcgt_4,170,145). coords(fcgt_4,101,210,145).  visible(fcgt_4,145). happensAt(inactive(fcgt_4),145).
orientation(fcgt_3,159,146). coords(fcgt_3,103,163,146).  visible(fcgt_3,146). happensAt(inactive(fcgt_3),146). orientation(fcgt_4,170,146). coords(fcgt_4,101,210,146).  visible(fcgt_4,146). happensAt(inactive(fcgt_4),146).
orientation(fcgt_3,159,147). coords(fcgt_3,103,163,147).  visible(fcgt_3,147). happensAt(inactive(fcgt_3),147). orientation(fcgt_4,170,147). coords(fcgt_4,101,210,147).  visible(fcgt_4,147). happensAt(inactive(fcgt_4),147).
orientation(fcgt_3,159,148). coords(fcgt_3,103,163,148).  visible(fcgt_3,148). happensAt(inactive(fcgt_3),148). orientation(fcgt_4,170,148). coords(fcgt_4,101,210,148).  visible(fcgt_4,148). happensAt(inactive(fcgt_4),148).
orientation(fcgt_3,159,149). coords(fcgt_3,103,163,149).  visible(fcgt_3,149). happensAt(inactive(fcgt_3),149). orientation(fcgt_4,170,149). coords(fcgt_4,101,210,149).  visible(fcgt_4,149). happensAt(inactive(fcgt_4),149).
orientation(fcgt_3,159,150). coords(fcgt_3,103,163,150).  visible(fcgt_3,150). happensAt(inactive(fcgt_3),150). orientation(fcgt_4,170,150). coords(fcgt_4,101,210,150).  visible(fcgt_4,150). happensAt(walking(fcgt_4),150).
orientation(fcgt_3,159,151). coords(fcgt_3,103,163,151).  visible(fcgt_3,151). happensAt(inactive(fcgt_3),151). orientation(fcgt_4,170,151). coords(fcgt_4,101,210,151).  visible(fcgt_4,151). happensAt(walking(fcgt_4),151).
orientation(fcgt_3,159,152). coords(fcgt_3,103,163,152).  visible(fcgt_3,152). happensAt(inactive(fcgt_3),152). orientation(fcgt_4,170,152). coords(fcgt_4,101,210,152).  visible(fcgt_4,152). happensAt(walking(fcgt_4),152).
orientation(fcgt_3,159,153). coords(fcgt_3,103,163,153).  visible(fcgt_3,153). happensAt(inactive(fcgt_3),153). orientation(fcgt_4,170,153). coords(fcgt_4,101,210,153).  visible(fcgt_4,153). happensAt(walking(fcgt_4),153).
orientation(fcgt_3,159,154). coords(fcgt_3,103,163,154).  visible(fcgt_3,154). happensAt(inactive(fcgt_3),154). orientation(fcgt_4,170,154). coords(fcgt_4,101,210,154).  visible(fcgt_4,154). happensAt(walking(fcgt_4),154).
orientation(fcgt_3,159,155). coords(fcgt_3,103,163,155).  visible(fcgt_3,155). happensAt(inactive(fcgt_3),155). orientation(fcgt_4,170,155). coords(fcgt_4,100,210,155).  visible(fcgt_4,155). happensAt(walking(fcgt_4),155).
orientation(fcgt_3,159,156). coords(fcgt_3,103,163,156).  visible(fcgt_3,156). happensAt(inactive(fcgt_3),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
orientation(fcgt_3,159,157). coords(fcgt_3,103,163,157).  visible(fcgt_3,157). happensAt(inactive(fcgt_3),157). orientation(fcgt_4,170,157). coords(fcgt_4,100,210,157).  visible(fcgt_4,157). happensAt(walking(fcgt_4),157).
orientation(fcgt_3,159,158). coords(fcgt_3,103,163,158).  visible(fcgt_3,158). happensAt(inactive(fcgt_3),158). orientation(fcgt_4,170,158). coords(fcgt_4,100,210,158).  visible(fcgt_4,158). happensAt(walking(fcgt_4),158).
orientation(fcgt_3,159,159). coords(fcgt_3,103,163,159).  visible(fcgt_3,159). happensAt(inactive(fcgt_3),159). orientation(fcgt_4,170,159). coords(fcgt_4,100,210,159).  visible(fcgt_4,159). happensAt(walking(fcgt_4),159).
orientation(fcgt_3,159,160). coords(fcgt_3,103,163,160).  visible(fcgt_3,160). happensAt(inactive(fcgt_3),160). orientation(fcgt_4,170,160). coords(fcgt_4,100,210,160).  visible(fcgt_4,160). happensAt(walking(fcgt_4),160).
orientation(fcgt_3,159,161). coords(fcgt_3,103,163,161).  visible(fcgt_3,161). happensAt(inactive(fcgt_3),161). orientation(fcgt_4,170,161). coords(fcgt_4,100,210,161).  visible(fcgt_4,161). happensAt(walking(fcgt_4),161).
orientation(fcgt_3,159,162). coords(fcgt_3,103,163,162).  visible(fcgt_3,162). happensAt(inactive(fcgt_3),162). orientation(fcgt_4,165,162). coords(fcgt_4,99,210,162).  visible(fcgt_4,162). happensAt(walking(fcgt_4),162).
orientation(fcgt_3,159,163). coords(fcgt_3,103,163,163).  visible(fcgt_3,163). happensAt(inactive(fcgt_3),163). orientation(fcgt_4,165,163). coords(fcgt_4,99,210,163).  visible(fcgt_4,163). happensAt(walking(fcgt_4),163).
orientation(fcgt_3,159,164). coords(fcgt_3,103,163,164).  visible(fcgt_3,164). happensAt(inactive(fcgt_3),164). orientation(fcgt_4,165,164). coords(fcgt_4,99,210,164).  visible(fcgt_4,164). happensAt(walking(fcgt_4),164).
orientation(fcgt_3,159,165). coords(fcgt_3,103,163,165).  visible(fcgt_3,165). happensAt(inactive(fcgt_3),165). orientation(fcgt_4,164,165). coords(fcgt_4,99,209,165).  visible(fcgt_4,165). happensAt(walking(fcgt_4),165).
orientation(fcgt_3,159,166). coords(fcgt_3,103,163,166).  visible(fcgt_3,166). happensAt(inactive(fcgt_3),166). orientation(fcgt_4,165,166). coords(fcgt_4,99,209,166).  visible(fcgt_4,166). happensAt(walking(fcgt_4),166).
orientation(fcgt_3,159,167). coords(fcgt_3,103,163,167).  visible(fcgt_3,167). happensAt(inactive(fcgt_3),167). orientation(fcgt_4,165,167). coords(fcgt_4,99,209,167).  visible(fcgt_4,167). happensAt(walking(fcgt_4),167).
orientation(fcgt_3,159,168). coords(fcgt_3,103,163,168).  visible(fcgt_3,168). happensAt(inactive(fcgt_3),168). orientation(fcgt_4,163,168). coords(fcgt_4,99,209,168).  visible(fcgt_4,168). happensAt(walking(fcgt_4),168).
#show holdsAt/2.