
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

	orientation(ricgt_4,172,143). coords(ricgt_4,26,136,143).  visible(ricgt_4,143). happensAt(walking(ricgt_4),143). orientation(ricgt_5,0,143). coords(ricgt_5,27,128,143).  visible(ricgt_5,143). happensAt(walking(ricgt_5),143).
orientation(ricgt_4,172,144). coords(ricgt_4,26,137,144).  visible(ricgt_4,144). happensAt(walking(ricgt_4),144). orientation(ricgt_5,0,144). coords(ricgt_5,27,128,144).  visible(ricgt_5,144). happensAt(walking(ricgt_5),144).
orientation(ricgt_4,172,145). coords(ricgt_4,25,137,145).  visible(ricgt_4,145). happensAt(walking(ricgt_4),145). orientation(ricgt_5,0,145). coords(ricgt_5,27,129,145).  visible(ricgt_5,145). happensAt(walking(ricgt_5),145).
orientation(ricgt_4,172,146). coords(ricgt_4,25,137,146).  visible(ricgt_4,146). happensAt(walking(ricgt_4),146). orientation(ricgt_5,0,146). coords(ricgt_5,27,129,146).  visible(ricgt_5,146). happensAt(walking(ricgt_5),146).
orientation(ricgt_4,172,147). coords(ricgt_4,25,137,147).  visible(ricgt_4,147). happensAt(walking(ricgt_4),147). orientation(ricgt_5,0,147). coords(ricgt_5,27,129,147).  visible(ricgt_5,147). happensAt(walking(ricgt_5),147).
orientation(ricgt_4,172,148). coords(ricgt_4,25,137,148).  visible(ricgt_4,148). happensAt(walking(ricgt_4),148). orientation(ricgt_5,0,148). coords(ricgt_5,27,129,148).  visible(ricgt_5,148). happensAt(walking(ricgt_5),148).
orientation(ricgt_4,172,149). coords(ricgt_4,25,137,149).  visible(ricgt_4,149). happensAt(walking(ricgt_4),149). orientation(ricgt_5,0,149). coords(ricgt_5,27,129,149).  visible(ricgt_5,149). happensAt(walking(ricgt_5),149).
orientation(ricgt_4,172,150). coords(ricgt_4,25,138,150).  visible(ricgt_4,150). happensAt(walking(ricgt_4),150). orientation(ricgt_5,0,150). coords(ricgt_5,27,130,150).  visible(ricgt_5,150). happensAt(walking(ricgt_5),150).
orientation(ricgt_4,172,151). coords(ricgt_4,25,138,151).  visible(ricgt_4,151). happensAt(walking(ricgt_4),151). orientation(ricgt_5,0,151). coords(ricgt_5,27,130,151).  visible(ricgt_5,151). happensAt(walking(ricgt_5),151).
orientation(ricgt_4,172,152). coords(ricgt_4,25,138,152).  visible(ricgt_4,152). happensAt(walking(ricgt_4),152). orientation(ricgt_5,0,152). coords(ricgt_5,27,130,152).  visible(ricgt_5,152). happensAt(walking(ricgt_5),152).
orientation(ricgt_4,172,153). coords(ricgt_4,25,138,153).  visible(ricgt_4,153). happensAt(walking(ricgt_4),153). orientation(ricgt_5,0,153). coords(ricgt_5,27,130,153).  visible(ricgt_5,153). happensAt(walking(ricgt_5),153).
orientation(ricgt_4,172,154). coords(ricgt_4,25,138,154).  visible(ricgt_4,154). happensAt(walking(ricgt_4),154). orientation(ricgt_5,0,154). coords(ricgt_5,27,131,154).  visible(ricgt_5,154). happensAt(walking(ricgt_5),154).
orientation(ricgt_4,172,155). coords(ricgt_4,25,138,155).  visible(ricgt_4,155). happensAt(walking(ricgt_4),155). orientation(ricgt_5,0,155). coords(ricgt_5,27,131,155).  visible(ricgt_5,155). happensAt(walking(ricgt_5),155).
orientation(ricgt_4,172,156). coords(ricgt_4,25,138,156).  visible(ricgt_4,156). happensAt(walking(ricgt_4),156). orientation(ricgt_5,0,156). coords(ricgt_5,27,132,156).  visible(ricgt_5,156). happensAt(walking(ricgt_5),156).
orientation(ricgt_4,172,157). coords(ricgt_4,25,138,157).  visible(ricgt_4,157). happensAt(walking(ricgt_4),157). orientation(ricgt_5,0,157). coords(ricgt_5,27,132,157).  visible(ricgt_5,157). happensAt(walking(ricgt_5),157).
orientation(ricgt_4,172,158). coords(ricgt_4,25,138,158).  visible(ricgt_4,158). happensAt(walking(ricgt_4),158). orientation(ricgt_5,0,158). coords(ricgt_5,27,132,158).  visible(ricgt_5,158). happensAt(walking(ricgt_5),158).
orientation(ricgt_4,172,159). coords(ricgt_4,25,139,159).  visible(ricgt_4,159). happensAt(walking(ricgt_4),159). orientation(ricgt_5,0,159). coords(ricgt_5,27,132,159).  visible(ricgt_5,159). happensAt(walking(ricgt_5),159).
orientation(ricgt_4,172,160). coords(ricgt_4,25,139,160).  visible(ricgt_4,160). happensAt(walking(ricgt_4),160). orientation(ricgt_5,0,160). coords(ricgt_5,27,133,160).  visible(ricgt_5,160). happensAt(walking(ricgt_5),160).
orientation(ricgt_4,172,161). coords(ricgt_4,25,139,161).  visible(ricgt_4,161). happensAt(walking(ricgt_4),161). orientation(ricgt_5,0,161). coords(ricgt_5,27,133,161).  visible(ricgt_5,161). happensAt(walking(ricgt_5),161).
orientation(ricgt_4,172,162). coords(ricgt_4,25,139,162).  visible(ricgt_4,162). happensAt(walking(ricgt_4),162). orientation(ricgt_5,0,162). coords(ricgt_5,27,133,162).  visible(ricgt_5,162). happensAt(walking(ricgt_5),162).
orientation(ricgt_4,172,163). coords(ricgt_4,25,139,163).  visible(ricgt_4,163). happensAt(walking(ricgt_4),163). orientation(ricgt_5,0,163). coords(ricgt_5,27,133,163).  visible(ricgt_5,163). happensAt(walking(ricgt_5),163).
orientation(ricgt_4,172,164). coords(ricgt_4,25,139,164).  visible(ricgt_4,164). happensAt(walking(ricgt_4),164). orientation(ricgt_5,0,164). coords(ricgt_5,27,133,164).  visible(ricgt_5,164). happensAt(walking(ricgt_5),164).
orientation(ricgt_4,172,165). coords(ricgt_4,25,140,165).  visible(ricgt_4,165). happensAt(walking(ricgt_4),165). orientation(ricgt_5,0,165). coords(ricgt_5,27,133,165).  visible(ricgt_5,165). happensAt(walking(ricgt_5),165).
orientation(ricgt_4,172,166). coords(ricgt_4,25,140,166).  visible(ricgt_4,166). happensAt(walking(ricgt_4),166). orientation(ricgt_5,0,166). coords(ricgt_5,27,133,166).  visible(ricgt_5,166). happensAt(walking(ricgt_5),166).
orientation(ricgt_4,172,167). coords(ricgt_4,25,140,167).  visible(ricgt_4,167). happensAt(walking(ricgt_4),167). orientation(ricgt_5,0,167). coords(ricgt_5,27,133,167).  visible(ricgt_5,167). happensAt(walking(ricgt_5),167).
#show holdsAt/2.