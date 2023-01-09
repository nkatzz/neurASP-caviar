
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

	orientation(rffgt_0,135,144). coords(rffgt_0,158,32,144).  visible(rffgt_0,144). happensAt(walking(rffgt_0),144). orientation(rffgt_1,148,144). coords(rffgt_1,152,109,144).  visible(rffgt_1,144). happensAt(inactive(rffgt_1),144).
orientation(rffgt_0,135,145). coords(rffgt_0,159,31,145).  visible(rffgt_0,145). happensAt(walking(rffgt_0),145). orientation(rffgt_1,148,145). coords(rffgt_1,152,109,145).  visible(rffgt_1,145). happensAt(inactive(rffgt_1),145).
orientation(rffgt_0,135,146). coords(rffgt_0,158,30,146).  visible(rffgt_0,146). happensAt(walking(rffgt_0),146). orientation(rffgt_1,148,146). coords(rffgt_1,152,109,146).  visible(rffgt_1,146). happensAt(inactive(rffgt_1),146).
orientation(rffgt_0,135,147). coords(rffgt_0,159,30,147).  visible(rffgt_0,147). happensAt(walking(rffgt_0),147). orientation(rffgt_1,148,147). coords(rffgt_1,152,109,147).  visible(rffgt_1,147). happensAt(inactive(rffgt_1),147).
orientation(rffgt_0,135,148). coords(rffgt_0,160,30,148).  visible(rffgt_0,148). happensAt(walking(rffgt_0),148). orientation(rffgt_1,148,148). coords(rffgt_1,152,109,148).  visible(rffgt_1,148). happensAt(inactive(rffgt_1),148).
orientation(rffgt_0,135,149). coords(rffgt_0,162,30,149).  visible(rffgt_0,149). happensAt(walking(rffgt_0),149). orientation(rffgt_1,148,149). coords(rffgt_1,152,109,149).  visible(rffgt_1,149). happensAt(inactive(rffgt_1),149).
orientation(rffgt_0,135,150). coords(rffgt_0,163,29,150).  visible(rffgt_0,150). happensAt(walking(rffgt_0),150). orientation(rffgt_1,148,150). coords(rffgt_1,152,109,150).  visible(rffgt_1,150). happensAt(inactive(rffgt_1),150).
orientation(rffgt_0,135,151). coords(rffgt_0,163,29,151).  visible(rffgt_0,151). happensAt(walking(rffgt_0),151). orientation(rffgt_1,148,151). coords(rffgt_1,152,109,151).  visible(rffgt_1,151). happensAt(inactive(rffgt_1),151).
orientation(rffgt_0,135,152). coords(rffgt_0,164,29,152).  visible(rffgt_0,152). happensAt(walking(rffgt_0),152). orientation(rffgt_1,148,152). coords(rffgt_1,152,109,152).  visible(rffgt_1,152). happensAt(inactive(rffgt_1),152).
orientation(rffgt_0,135,153). coords(rffgt_0,164,29,153).  visible(rffgt_0,153). happensAt(walking(rffgt_0),153). orientation(rffgt_1,148,153). coords(rffgt_1,152,109,153).  visible(rffgt_1,153). happensAt(inactive(rffgt_1),153).
orientation(rffgt_0,135,154). coords(rffgt_0,164,28,154).  visible(rffgt_0,154). happensAt(walking(rffgt_0),154). orientation(rffgt_1,148,154). coords(rffgt_1,152,109,154).  visible(rffgt_1,154). happensAt(inactive(rffgt_1),154).
orientation(rffgt_0,135,155). coords(rffgt_0,164,28,155).  visible(rffgt_0,155). happensAt(walking(rffgt_0),155). orientation(rffgt_1,148,155). coords(rffgt_1,152,109,155).  visible(rffgt_1,155). happensAt(inactive(rffgt_1),155).
orientation(rffgt_0,131,156). coords(rffgt_0,165,28,156).  visible(rffgt_0,156). happensAt(walking(rffgt_0),156). orientation(rffgt_1,148,156). coords(rffgt_1,152,109,156).  visible(rffgt_1,156). happensAt(inactive(rffgt_1),156).
orientation(rffgt_0,131,157). coords(rffgt_0,165,28,157).  visible(rffgt_0,157). happensAt(walking(rffgt_0),157). orientation(rffgt_1,148,157). coords(rffgt_1,152,109,157).  visible(rffgt_1,157). happensAt(inactive(rffgt_1),157).
orientation(rffgt_0,131,158). coords(rffgt_0,165,27,158).  visible(rffgt_0,158). happensAt(walking(rffgt_0),158). orientation(rffgt_1,148,158). coords(rffgt_1,152,109,158).  visible(rffgt_1,158). happensAt(inactive(rffgt_1),158).
orientation(rffgt_0,131,159). coords(rffgt_0,166,27,159).  visible(rffgt_0,159). happensAt(walking(rffgt_0),159). orientation(rffgt_1,148,159). coords(rffgt_1,152,109,159).  visible(rffgt_1,159). happensAt(inactive(rffgt_1),159).
orientation(rffgt_0,131,160). coords(rffgt_0,166,26,160).  visible(rffgt_0,160). happensAt(walking(rffgt_0),160). orientation(rffgt_1,148,160). coords(rffgt_1,152,109,160).  visible(rffgt_1,160). happensAt(inactive(rffgt_1),160).
orientation(rffgt_0,131,161). coords(rffgt_0,166,26,161).  visible(rffgt_0,161). happensAt(walking(rffgt_0),161). orientation(rffgt_1,148,161). coords(rffgt_1,152,109,161).  visible(rffgt_1,161). happensAt(inactive(rffgt_1),161).
orientation(rffgt_0,131,162). coords(rffgt_0,166,26,162).  visible(rffgt_0,162). happensAt(walking(rffgt_0),162). orientation(rffgt_1,148,162). coords(rffgt_1,152,109,162).  visible(rffgt_1,162). happensAt(inactive(rffgt_1),162).
orientation(rffgt_0,131,163). coords(rffgt_0,167,25,163).  visible(rffgt_0,163). happensAt(walking(rffgt_0),163). orientation(rffgt_1,148,163). coords(rffgt_1,152,109,163).  visible(rffgt_1,163). happensAt(inactive(rffgt_1),163).
orientation(rffgt_0,131,164). coords(rffgt_0,166,24,164).  visible(rffgt_0,164). happensAt(walking(rffgt_0),164). orientation(rffgt_1,148,164). coords(rffgt_1,152,109,164).  visible(rffgt_1,164). happensAt(inactive(rffgt_1),164).
orientation(rffgt_0,131,165). coords(rffgt_0,167,24,165).  visible(rffgt_0,165). happensAt(walking(rffgt_0),165). orientation(rffgt_1,148,165). coords(rffgt_1,152,109,165).  visible(rffgt_1,165). happensAt(inactive(rffgt_1),165).
orientation(rffgt_0,131,166). coords(rffgt_0,167,23,166).  visible(rffgt_0,166). happensAt(walking(rffgt_0),166). orientation(rffgt_1,148,166). coords(rffgt_1,152,109,166).  visible(rffgt_1,166). happensAt(inactive(rffgt_1),166).
orientation(rffgt_0,131,167). coords(rffgt_0,167,22,167).  visible(rffgt_0,167). happensAt(walking(rffgt_0),167). orientation(rffgt_1,148,167). coords(rffgt_1,152,109,167).  visible(rffgt_1,167). happensAt(inactive(rffgt_1),167).
orientation(rffgt_0,131,168). coords(rffgt_0,167,21,168).  visible(rffgt_0,168). happensAt(walking(rffgt_0),168). orientation(rffgt_1,148,168). coords(rffgt_1,152,109,168).  visible(rffgt_1,168). happensAt(inactive(rffgt_1),168).
#show holdsAt/2.