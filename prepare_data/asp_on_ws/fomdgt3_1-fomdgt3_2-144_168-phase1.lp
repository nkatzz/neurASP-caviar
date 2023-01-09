
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

	orientation(fomdgt3_1,150,144). coords(fomdgt3_1,113,108,144).  visible(fomdgt3_1,144). happensAt(walking(fomdgt3_1),144). orientation(fomdgt3_2,5,144). coords(fomdgt3_2,268,244,144).  visible(fomdgt3_2,144). happensAt(walking(fomdgt3_2),144).
orientation(fomdgt3_1,150,145). coords(fomdgt3_1,110,108,145).  visible(fomdgt3_1,145). happensAt(walking(fomdgt3_1),145). orientation(fomdgt3_2,5,145). coords(fomdgt3_2,268,244,145).  visible(fomdgt3_2,145). happensAt(walking(fomdgt3_2),145).
orientation(fomdgt3_1,150,146). coords(fomdgt3_1,110,108,146).  visible(fomdgt3_1,146). happensAt(walking(fomdgt3_1),146). orientation(fomdgt3_2,10,146). coords(fomdgt3_2,268,238,146).  visible(fomdgt3_2,146). happensAt(walking(fomdgt3_2),146).
orientation(fomdgt3_1,150,147). coords(fomdgt3_1,110,108,147).  visible(fomdgt3_1,147). happensAt(walking(fomdgt3_1),147). orientation(fomdgt3_2,10,147). coords(fomdgt3_2,265,235,147).  visible(fomdgt3_2,147). happensAt(walking(fomdgt3_2),147).
orientation(fomdgt3_1,150,148). coords(fomdgt3_1,110,108,148).  visible(fomdgt3_1,148). happensAt(walking(fomdgt3_1),148). orientation(fomdgt3_2,10,148). coords(fomdgt3_2,265,232,148).  visible(fomdgt3_2,148). happensAt(walking(fomdgt3_2),148).
orientation(fomdgt3_1,150,149). coords(fomdgt3_1,110,108,149).  visible(fomdgt3_1,149). happensAt(walking(fomdgt3_1),149). orientation(fomdgt3_2,10,149). coords(fomdgt3_2,262,230,149).  visible(fomdgt3_2,149). happensAt(walking(fomdgt3_2),149).
orientation(fomdgt3_1,150,150). coords(fomdgt3_1,107,108,150).  visible(fomdgt3_1,150). happensAt(walking(fomdgt3_1),150). orientation(fomdgt3_2,10,150). coords(fomdgt3_2,262,228,150).  visible(fomdgt3_2,150). happensAt(walking(fomdgt3_2),150).
orientation(fomdgt3_1,150,151). coords(fomdgt3_1,107,108,151).  visible(fomdgt3_1,151). happensAt(walking(fomdgt3_1),151). orientation(fomdgt3_2,10,151). coords(fomdgt3_2,260,226,151).  visible(fomdgt3_2,151). happensAt(walking(fomdgt3_2),151).
orientation(fomdgt3_1,150,152). coords(fomdgt3_1,106,108,152).  visible(fomdgt3_1,152). happensAt(walking(fomdgt3_1),152). orientation(fomdgt3_2,18,152). coords(fomdgt3_2,260,223,152).  visible(fomdgt3_2,152). happensAt(walking(fomdgt3_2),152).
orientation(fomdgt3_1,150,153). coords(fomdgt3_1,106,108,153).  visible(fomdgt3_1,153). happensAt(walking(fomdgt3_1),153). orientation(fomdgt3_2,18,153). coords(fomdgt3_2,260,221,153).  visible(fomdgt3_2,153). happensAt(walking(fomdgt3_2),153).
orientation(fomdgt3_1,150,154). coords(fomdgt3_1,106,108,154).  visible(fomdgt3_1,154). happensAt(walking(fomdgt3_1),154). orientation(fomdgt3_2,18,154). coords(fomdgt3_2,260,221,154).  visible(fomdgt3_2,154). happensAt(walking(fomdgt3_2),154).
orientation(fomdgt3_1,150,155). coords(fomdgt3_1,106,108,155).  visible(fomdgt3_1,155). happensAt(walking(fomdgt3_1),155). orientation(fomdgt3_2,18,155). coords(fomdgt3_2,260,221,155).  visible(fomdgt3_2,155). happensAt(walking(fomdgt3_2),155).
orientation(fomdgt3_1,150,156). coords(fomdgt3_1,106,108,156).  visible(fomdgt3_1,156). happensAt(walking(fomdgt3_1),156). orientation(fomdgt3_2,18,156). coords(fomdgt3_2,260,221,156).  visible(fomdgt3_2,156). happensAt(walking(fomdgt3_2),156).
orientation(fomdgt3_1,150,157). coords(fomdgt3_1,99,108,157).  visible(fomdgt3_1,157). happensAt(walking(fomdgt3_1),157). orientation(fomdgt3_2,18,157). coords(fomdgt3_2,260,221,157).  visible(fomdgt3_2,157). happensAt(walking(fomdgt3_2),157).
orientation(fomdgt3_1,150,158). coords(fomdgt3_1,98,108,158).  visible(fomdgt3_1,158). happensAt(walking(fomdgt3_1),158). orientation(fomdgt3_2,170,158). coords(fomdgt3_2,260,218,158).  visible(fomdgt3_2,158). happensAt(walking(fomdgt3_2),158).
orientation(fomdgt3_1,150,159). coords(fomdgt3_1,98,108,159).  visible(fomdgt3_1,159). happensAt(walking(fomdgt3_1),159). orientation(fomdgt3_2,160,159). coords(fomdgt3_2,260,215,159).  visible(fomdgt3_2,159). happensAt(walking(fomdgt3_2),159).
orientation(fomdgt3_1,150,160). coords(fomdgt3_1,98,108,160).  visible(fomdgt3_1,160). happensAt(walking(fomdgt3_1),160). orientation(fomdgt3_2,160,160). coords(fomdgt3_2,257,214,160).  visible(fomdgt3_2,160). happensAt(walking(fomdgt3_2),160).
orientation(fomdgt3_1,150,161). coords(fomdgt3_1,97,108,161).  visible(fomdgt3_1,161). happensAt(walking(fomdgt3_1),161). orientation(fomdgt3_2,160,161). coords(fomdgt3_2,257,210,161).  visible(fomdgt3_2,161). happensAt(walking(fomdgt3_2),161).
orientation(fomdgt3_1,150,162). coords(fomdgt3_1,97,108,162).  visible(fomdgt3_1,162). happensAt(walking(fomdgt3_1),162). orientation(fomdgt3_2,160,162). coords(fomdgt3_2,254,207,162).  visible(fomdgt3_2,162). happensAt(walking(fomdgt3_2),162).
orientation(fomdgt3_1,150,163). coords(fomdgt3_1,97,108,163).  visible(fomdgt3_1,163). happensAt(walking(fomdgt3_1),163). orientation(fomdgt3_2,160,163). coords(fomdgt3_2,254,203,163).  visible(fomdgt3_2,163). happensAt(walking(fomdgt3_2),163).
orientation(fomdgt3_1,150,164). coords(fomdgt3_1,97,108,164).  visible(fomdgt3_1,164). happensAt(walking(fomdgt3_1),164). orientation(fomdgt3_2,160,164). coords(fomdgt3_2,253,200,164).  visible(fomdgt3_2,164). happensAt(walking(fomdgt3_2),164).
orientation(fomdgt3_1,150,165). coords(fomdgt3_1,97,108,165).  visible(fomdgt3_1,165). happensAt(walking(fomdgt3_1),165). orientation(fomdgt3_2,160,165). coords(fomdgt3_2,252,199,165).  visible(fomdgt3_2,165). happensAt(walking(fomdgt3_2),165).
orientation(fomdgt3_1,150,166). coords(fomdgt3_1,96,108,166).  visible(fomdgt3_1,166). happensAt(walking(fomdgt3_1),166). orientation(fomdgt3_2,160,166). coords(fomdgt3_2,252,196,166).  visible(fomdgt3_2,166). happensAt(walking(fomdgt3_2),166).
orientation(fomdgt3_1,150,167). coords(fomdgt3_1,96,108,167).  visible(fomdgt3_1,167). happensAt(walking(fomdgt3_1),167). orientation(fomdgt3_2,160,167). coords(fomdgt3_2,251,195,167).  visible(fomdgt3_2,167). happensAt(walking(fomdgt3_2),167).
orientation(fomdgt3_1,150,168). coords(fomdgt3_1,96,108,168).  visible(fomdgt3_1,168). happensAt(walking(fomdgt3_1),168). orientation(fomdgt3_2,160,168). coords(fomdgt3_2,251,195,168).  visible(fomdgt3_2,168). happensAt(walking(fomdgt3_2),168).
#show holdsAt/2.