
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

	orientation(lbgt_0,154,144). coords(lbgt_0,190,170,144).  visible(lbgt_0,144). happensAt(walking(lbgt_0),144). orientation(lbgt_1,153,144). coords(lbgt_1,77,92,144).  visible(lbgt_1,144). happensAt(walking(lbgt_1),144).
orientation(lbgt_0,154,145). coords(lbgt_0,192,174,145).  visible(lbgt_0,145). happensAt(walking(lbgt_0),145). orientation(lbgt_1,153,145). coords(lbgt_1,77,91,145).  visible(lbgt_1,145). happensAt(walking(lbgt_1),145).
orientation(lbgt_0,153,146). coords(lbgt_0,194,178,146).  visible(lbgt_0,146). happensAt(walking(lbgt_0),146). orientation(lbgt_1,153,146). coords(lbgt_1,77,92,146).  visible(lbgt_1,146). happensAt(walking(lbgt_1),146).
orientation(lbgt_0,153,147). coords(lbgt_0,195,180,147).  visible(lbgt_0,147). happensAt(walking(lbgt_0),147). orientation(lbgt_1,153,147). coords(lbgt_1,77,92,147).  visible(lbgt_1,147). happensAt(walking(lbgt_1),147).
orientation(lbgt_0,153,148). coords(lbgt_0,196,182,148).  visible(lbgt_0,148). happensAt(walking(lbgt_0),148). orientation(lbgt_1,153,148). coords(lbgt_1,77,92,148).  visible(lbgt_1,148). happensAt(walking(lbgt_1),148).
orientation(lbgt_0,153,149). coords(lbgt_0,197,184,149).  visible(lbgt_0,149). happensAt(walking(lbgt_0),149). orientation(lbgt_1,157,149). coords(lbgt_1,78,93,149).  visible(lbgt_1,149). happensAt(walking(lbgt_1),149).
orientation(lbgt_0,153,150). coords(lbgt_0,199,186,150).  visible(lbgt_0,150). happensAt(walking(lbgt_0),150). orientation(lbgt_1,157,150). coords(lbgt_1,78,93,150).  visible(lbgt_1,150). happensAt(walking(lbgt_1),150).
orientation(lbgt_0,153,151). coords(lbgt_0,201,187,151).  visible(lbgt_0,151). happensAt(walking(lbgt_0),151). orientation(lbgt_1,157,151). coords(lbgt_1,78,93,151).  visible(lbgt_1,151). happensAt(walking(lbgt_1),151).
orientation(lbgt_0,160,152). coords(lbgt_0,203,187,152).  visible(lbgt_0,152). happensAt(walking(lbgt_0),152). orientation(lbgt_1,157,152). coords(lbgt_1,78,93,152).  visible(lbgt_1,152). happensAt(walking(lbgt_1),152).
orientation(lbgt_0,166,153). coords(lbgt_0,206,189,153).  visible(lbgt_0,153). happensAt(walking(lbgt_0),153). orientation(lbgt_1,157,153). coords(lbgt_1,79,93,153).  visible(lbgt_1,153). happensAt(walking(lbgt_1),153).
orientation(lbgt_0,166,154). coords(lbgt_0,208,195,154).  visible(lbgt_0,154). happensAt(walking(lbgt_0),154). orientation(lbgt_1,157,154). coords(lbgt_1,79,94,154).  visible(lbgt_1,154). happensAt(walking(lbgt_1),154).
orientation(lbgt_0,161,155). coords(lbgt_0,209,198,155).  visible(lbgt_0,155). happensAt(walking(lbgt_0),155). orientation(lbgt_1,157,155). coords(lbgt_1,79,94,155).  visible(lbgt_1,155). happensAt(walking(lbgt_1),155).
orientation(lbgt_0,161,156). coords(lbgt_0,211,199,156).  visible(lbgt_0,156). happensAt(walking(lbgt_0),156). orientation(lbgt_1,157,156). coords(lbgt_1,79,94,156).  visible(lbgt_1,156). happensAt(walking(lbgt_1),156).
orientation(lbgt_0,161,157). coords(lbgt_0,213,200,157).  visible(lbgt_0,157). happensAt(walking(lbgt_0),157). orientation(lbgt_1,157,157). coords(lbgt_1,79,94,157).  visible(lbgt_1,157). happensAt(walking(lbgt_1),157).
orientation(lbgt_0,165,158). coords(lbgt_0,213,201,158).  visible(lbgt_0,158). happensAt(walking(lbgt_0),158). orientation(lbgt_1,157,158). coords(lbgt_1,79,94,158).  visible(lbgt_1,158). happensAt(walking(lbgt_1),158).
orientation(lbgt_0,165,159). coords(lbgt_0,214,201,159).  visible(lbgt_0,159). happensAt(active(lbgt_0),159). orientation(lbgt_1,157,159). coords(lbgt_1,79,94,159).  visible(lbgt_1,159). happensAt(walking(lbgt_1),159).
orientation(lbgt_0,165,160). coords(lbgt_0,212,202,160).  visible(lbgt_0,160). happensAt(active(lbgt_0),160). orientation(lbgt_1,157,160). coords(lbgt_1,79,95,160).  visible(lbgt_1,160). happensAt(walking(lbgt_1),160).
orientation(lbgt_0,165,161). coords(lbgt_0,211,202,161).  visible(lbgt_0,161). happensAt(active(lbgt_0),161). orientation(lbgt_1,157,161). coords(lbgt_1,79,97,161).  visible(lbgt_1,161). happensAt(walking(lbgt_1),161).
orientation(lbgt_0,168,162). coords(lbgt_0,209,201,162).  visible(lbgt_0,162). happensAt(active(lbgt_0),162). orientation(lbgt_1,157,162). coords(lbgt_1,79,97,162).  visible(lbgt_1,162). happensAt(walking(lbgt_1),162).
orientation(lbgt_0,168,163). coords(lbgt_0,207,201,163).  visible(lbgt_0,163). happensAt(active(lbgt_0),163). orientation(lbgt_1,157,163). coords(lbgt_1,79,97,163).  visible(lbgt_1,163). happensAt(walking(lbgt_1),163).
orientation(lbgt_0,168,164). coords(lbgt_0,205,201,164).  visible(lbgt_0,164). happensAt(active(lbgt_0),164). orientation(lbgt_1,152,164). coords(lbgt_1,79,97,164).  visible(lbgt_1,164). happensAt(walking(lbgt_1),164).
orientation(lbgt_0,168,165). coords(lbgt_0,205,201,165).  visible(lbgt_0,165). happensAt(active(lbgt_0),165). orientation(lbgt_1,152,165). coords(lbgt_1,79,98,165).  visible(lbgt_1,165). happensAt(walking(lbgt_1),165).
orientation(lbgt_0,168,166). coords(lbgt_0,205,201,166).  visible(lbgt_0,166). happensAt(active(lbgt_0),166). orientation(lbgt_1,152,166). coords(lbgt_1,79,98,166).  visible(lbgt_1,166). happensAt(walking(lbgt_1),166).
orientation(lbgt_0,168,167). coords(lbgt_0,206,202,167).  visible(lbgt_0,167). happensAt(active(lbgt_0),167). orientation(lbgt_1,152,167). coords(lbgt_1,79,100,167).  visible(lbgt_1,167). happensAt(walking(lbgt_1),167).
orientation(lbgt_0,168,168). coords(lbgt_0,206,202,168).  visible(lbgt_0,168). happensAt(active(lbgt_0),168). orientation(lbgt_1,152,168). coords(lbgt_1,79,98,168).  visible(lbgt_1,168). happensAt(walking(lbgt_1),168).
#show holdsAt/2.