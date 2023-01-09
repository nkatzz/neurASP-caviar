
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

	orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). happensAt(walking(mc1gt_0),141). orientation(mc1gt_2,144,141). coords(mc1gt_2,75,93,141).  visible(mc1gt_2,141). happensAt(walking(mc1gt_2),141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). happensAt(walking(mc1gt_0),142). orientation(mc1gt_2,144,142). coords(mc1gt_2,75,94,142).  visible(mc1gt_2,142). happensAt(walking(mc1gt_2),142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). happensAt(walking(mc1gt_0),143). orientation(mc1gt_2,144,143). coords(mc1gt_2,73,95,143).  visible(mc1gt_2,143). happensAt(walking(mc1gt_2),143).
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). happensAt(walking(mc1gt_0),144). orientation(mc1gt_2,144,144). coords(mc1gt_2,73,95,144).  visible(mc1gt_2,144). happensAt(walking(mc1gt_2),144).
orientation(mc1gt_0,147,145). coords(mc1gt_0,95,107,145).  visible(mc1gt_0,145). happensAt(walking(mc1gt_0),145). orientation(mc1gt_2,144,145). coords(mc1gt_2,74,95,145).  visible(mc1gt_2,145). happensAt(walking(mc1gt_2),145).
orientation(mc1gt_0,147,146). coords(mc1gt_0,96,107,146).  visible(mc1gt_0,146). happensAt(walking(mc1gt_0),146). orientation(mc1gt_2,144,146). coords(mc1gt_2,74,95,146).  visible(mc1gt_2,146). happensAt(walking(mc1gt_2),146).
orientation(mc1gt_0,147,147). coords(mc1gt_0,98,106,147).  visible(mc1gt_0,147). happensAt(walking(mc1gt_0),147). orientation(mc1gt_2,144,147). coords(mc1gt_2,74,96,147).  visible(mc1gt_2,147). happensAt(walking(mc1gt_2),147).
orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). happensAt(walking(mc1gt_0),148). orientation(mc1gt_2,144,148). coords(mc1gt_2,75,96,148).  visible(mc1gt_2,148). happensAt(walking(mc1gt_2),148).
orientation(mc1gt_0,147,149). coords(mc1gt_0,101,107,149).  visible(mc1gt_0,149). happensAt(walking(mc1gt_0),149). orientation(mc1gt_2,144,149). coords(mc1gt_2,75,98,149).  visible(mc1gt_2,149). happensAt(walking(mc1gt_2),149).
orientation(mc1gt_0,147,150). coords(mc1gt_0,101,107,150).  visible(mc1gt_0,150). happensAt(walking(mc1gt_0),150). orientation(mc1gt_2,144,150). coords(mc1gt_2,78,101,150).  visible(mc1gt_2,150). happensAt(walking(mc1gt_2),150).
orientation(mc1gt_0,147,151). coords(mc1gt_0,102,107,151).  visible(mc1gt_0,151). happensAt(walking(mc1gt_0),151). orientation(mc1gt_2,144,151). coords(mc1gt_2,78,102,151).  visible(mc1gt_2,151). happensAt(walking(mc1gt_2),151).
orientation(mc1gt_0,147,152). coords(mc1gt_0,103,107,152).  visible(mc1gt_0,152). happensAt(walking(mc1gt_0),152). orientation(mc1gt_2,144,152). coords(mc1gt_2,79,103,152).  visible(mc1gt_2,152). happensAt(walking(mc1gt_2),152).
orientation(mc1gt_0,147,153). coords(mc1gt_0,104,107,153).  visible(mc1gt_0,153). happensAt(walking(mc1gt_0),153). orientation(mc1gt_2,144,153). coords(mc1gt_2,79,104,153).  visible(mc1gt_2,153). happensAt(walking(mc1gt_2),153).
orientation(mc1gt_0,147,154). coords(mc1gt_0,104,107,154).  visible(mc1gt_0,154). happensAt(walking(mc1gt_0),154). orientation(mc1gt_2,144,154). coords(mc1gt_2,80,105,154).  visible(mc1gt_2,154). happensAt(walking(mc1gt_2),154).
orientation(mc1gt_0,147,155). coords(mc1gt_0,104,107,155).  visible(mc1gt_0,155). happensAt(walking(mc1gt_0),155). orientation(mc1gt_2,144,155). coords(mc1gt_2,80,105,155).  visible(mc1gt_2,155). happensAt(walking(mc1gt_2),155).
orientation(mc1gt_0,147,156). coords(mc1gt_0,104,107,156).  visible(mc1gt_0,156). happensAt(walking(mc1gt_0),156). orientation(mc1gt_2,144,156). coords(mc1gt_2,80,105,156).  visible(mc1gt_2,156). happensAt(walking(mc1gt_2),156).
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). happensAt(walking(mc1gt_0),157). orientation(mc1gt_2,144,157). coords(mc1gt_2,80,105,157).  visible(mc1gt_2,157). happensAt(walking(mc1gt_2),157).
orientation(mc1gt_0,147,158). coords(mc1gt_0,104,108,158).  visible(mc1gt_0,158). happensAt(walking(mc1gt_0),158). orientation(mc1gt_2,144,158). coords(mc1gt_2,80,106,158).  visible(mc1gt_2,158). happensAt(walking(mc1gt_2),158).
orientation(mc1gt_0,147,159). coords(mc1gt_0,105,108,159).  visible(mc1gt_0,159). happensAt(walking(mc1gt_0),159). orientation(mc1gt_2,144,159). coords(mc1gt_2,80,106,159).  visible(mc1gt_2,159). happensAt(walking(mc1gt_2),159).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). happensAt(walking(mc1gt_0),160). orientation(mc1gt_2,144,160). coords(mc1gt_2,80,106,160).  visible(mc1gt_2,160). happensAt(walking(mc1gt_2),160).
orientation(mc1gt_0,147,161). coords(mc1gt_0,107,109,161).  visible(mc1gt_0,161). happensAt(walking(mc1gt_0),161). orientation(mc1gt_2,144,161). coords(mc1gt_2,81,106,161).  visible(mc1gt_2,161). happensAt(walking(mc1gt_2),161).
orientation(mc1gt_0,147,162). coords(mc1gt_0,108,109,162).  visible(mc1gt_0,162). happensAt(walking(mc1gt_0),162). orientation(mc1gt_2,144,162). coords(mc1gt_2,81,106,162).  visible(mc1gt_2,162). happensAt(walking(mc1gt_2),162).
orientation(mc1gt_0,147,163). coords(mc1gt_0,109,110,163).  visible(mc1gt_0,163). happensAt(walking(mc1gt_0),163). orientation(mc1gt_2,144,163). coords(mc1gt_2,82,107,163).  visible(mc1gt_2,163). happensAt(walking(mc1gt_2),163).
orientation(mc1gt_0,147,164). coords(mc1gt_0,110,110,164).  visible(mc1gt_0,164). happensAt(walking(mc1gt_0),164). orientation(mc1gt_2,144,164). coords(mc1gt_2,83,107,164).  visible(mc1gt_2,164). happensAt(walking(mc1gt_2),164).
orientation(mc1gt_0,147,165). coords(mc1gt_0,111,110,165).  visible(mc1gt_0,165). happensAt(walking(mc1gt_0),165). orientation(mc1gt_2,144,165). coords(mc1gt_2,84,107,165).  visible(mc1gt_2,165). happensAt(walking(mc1gt_2),165).
#show holdsAt/2.