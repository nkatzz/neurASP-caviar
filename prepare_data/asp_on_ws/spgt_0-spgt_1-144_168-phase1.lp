
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

	orientation(spgt_0,155,144). coords(spgt_0,103,128,144).  visible(spgt_0,144). happensAt(walking(spgt_0),144). orientation(spgt_1,148,144). coords(spgt_1,99,113,144).  visible(spgt_1,144). happensAt(walking(spgt_1),144).
orientation(spgt_0,159,145). coords(spgt_0,104,127,145).  visible(spgt_0,145). happensAt(walking(spgt_0),145). orientation(spgt_1,148,145). coords(spgt_1,99,113,145).  visible(spgt_1,145). happensAt(walking(spgt_1),145).
orientation(spgt_0,156,146). coords(spgt_0,104,127,146).  visible(spgt_0,146). happensAt(walking(spgt_0),146). orientation(spgt_1,148,146). coords(spgt_1,100,113,146).  visible(spgt_1,146). happensAt(walking(spgt_1),146).
orientation(spgt_0,154,147). coords(spgt_0,105,126,147).  visible(spgt_0,147). happensAt(walking(spgt_0),147). orientation(spgt_1,148,147). coords(spgt_1,101,113,147).  visible(spgt_1,147). happensAt(walking(spgt_1),147).
orientation(spgt_0,154,148). coords(spgt_0,105,126,148).  visible(spgt_0,148). happensAt(walking(spgt_0),148). orientation(spgt_1,148,148). coords(spgt_1,101,112,148).  visible(spgt_1,148). happensAt(walking(spgt_1),148).
orientation(spgt_0,154,149). coords(spgt_0,105,126,149).  visible(spgt_0,149). happensAt(walking(spgt_0),149). orientation(spgt_1,148,149). coords(spgt_1,102,112,149).  visible(spgt_1,149). happensAt(walking(spgt_1),149).
orientation(spgt_0,154,150). coords(spgt_0,106,125,150).  visible(spgt_0,150). happensAt(walking(spgt_0),150). orientation(spgt_1,148,150). coords(spgt_1,102,112,150).  visible(spgt_1,150). happensAt(walking(spgt_1),150).
orientation(spgt_0,154,151). coords(spgt_0,106,125,151).  visible(spgt_0,151). happensAt(walking(spgt_0),151). orientation(spgt_1,148,151). coords(spgt_1,102,111,151).  visible(spgt_1,151). happensAt(walking(spgt_1),151).
orientation(spgt_0,154,152). coords(spgt_0,106,124,152).  visible(spgt_0,152). happensAt(walking(spgt_0),152). orientation(spgt_1,148,152). coords(spgt_1,102,111,152).  visible(spgt_1,152). happensAt(walking(spgt_1),152).
orientation(spgt_0,154,153). coords(spgt_0,106,124,153).  visible(spgt_0,153). happensAt(walking(spgt_0),153). orientation(spgt_1,148,153). coords(spgt_1,103,110,153).  visible(spgt_1,153). happensAt(walking(spgt_1),153).
orientation(spgt_0,154,154). coords(spgt_0,106,123,154).  visible(spgt_0,154). happensAt(walking(spgt_0),154). orientation(spgt_1,148,154). coords(spgt_1,103,110,154).  visible(spgt_1,154). happensAt(walking(spgt_1),154).
orientation(spgt_0,154,155). coords(spgt_0,107,123,155).  visible(spgt_0,155). happensAt(walking(spgt_0),155). orientation(spgt_1,148,155). coords(spgt_1,103,110,155).  visible(spgt_1,155). happensAt(walking(spgt_1),155).
orientation(spgt_0,154,156). coords(spgt_0,107,123,156).  visible(spgt_0,156). happensAt(walking(spgt_0),156). orientation(spgt_1,148,156). coords(spgt_1,103,109,156).  visible(spgt_1,156). happensAt(walking(spgt_1),156).
orientation(spgt_0,154,157). coords(spgt_0,108,123,157).  visible(spgt_0,157). happensAt(walking(spgt_0),157). orientation(spgt_1,148,157). coords(spgt_1,104,109,157).  visible(spgt_1,157). happensAt(walking(spgt_1),157).
orientation(spgt_0,150,158). coords(spgt_0,108,123,158).  visible(spgt_0,158). happensAt(walking(spgt_0),158). orientation(spgt_1,148,158). coords(spgt_1,105,109,158).  visible(spgt_1,158). happensAt(walking(spgt_1),158).
orientation(spgt_0,150,159). coords(spgt_0,109,122,159).  visible(spgt_0,159). happensAt(walking(spgt_0),159). orientation(spgt_1,148,159). coords(spgt_1,106,108,159).  visible(spgt_1,159). happensAt(walking(spgt_1),159).
orientation(spgt_0,148,160). coords(spgt_0,109,122,160).  visible(spgt_0,160). happensAt(walking(spgt_0),160). orientation(spgt_1,148,160). coords(spgt_1,107,108,160).  visible(spgt_1,160). happensAt(walking(spgt_1),160).
orientation(spgt_0,148,161). coords(spgt_0,109,121,161).  visible(spgt_0,161). happensAt(walking(spgt_0),161). orientation(spgt_1,148,161). coords(spgt_1,107,108,161).  visible(spgt_1,161). happensAt(walking(spgt_1),161).
orientation(spgt_0,148,162). coords(spgt_0,109,121,162).  visible(spgt_0,162). happensAt(walking(spgt_0),162). orientation(spgt_1,148,162). coords(spgt_1,107,108,162).  visible(spgt_1,162). happensAt(walking(spgt_1),162).
orientation(spgt_0,148,163). coords(spgt_0,109,120,163).  visible(spgt_0,163). happensAt(walking(spgt_0),163). orientation(spgt_1,148,163). coords(spgt_1,107,107,163).  visible(spgt_1,163). happensAt(active(spgt_1),163).
orientation(spgt_0,148,164). coords(spgt_0,109,120,164).  visible(spgt_0,164). happensAt(walking(spgt_0),164). orientation(spgt_1,148,164). coords(spgt_1,107,106,164).  visible(spgt_1,164). happensAt(active(spgt_1),164).
orientation(spgt_0,148,165). coords(spgt_0,109,120,165).  visible(spgt_0,165). happensAt(walking(spgt_0),165). orientation(spgt_1,148,165). coords(spgt_1,107,105,165).  visible(spgt_1,165). happensAt(active(spgt_1),165).
orientation(spgt_0,148,166). coords(spgt_0,109,118,166).  visible(spgt_0,166). happensAt(walking(spgt_0),166). orientation(spgt_1,148,166). coords(spgt_1,107,105,166).  visible(spgt_1,166). happensAt(active(spgt_1),166).
orientation(spgt_0,148,167). coords(spgt_0,110,118,167).  visible(spgt_0,167). happensAt(walking(spgt_0),167). orientation(spgt_1,148,167). coords(spgt_1,108,105,167).  visible(spgt_1,167). happensAt(active(spgt_1),167).
orientation(spgt_0,148,168). coords(spgt_0,111,116,168).  visible(spgt_0,168). happensAt(walking(spgt_0),168). orientation(spgt_1,148,168). coords(spgt_1,108,104,168).  visible(spgt_1,168). happensAt(active(spgt_1),168).
#show holdsAt/2.