
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

	orientation(fra2gt_3,153,139). coords(fra2gt_3,29,207,139).  visible(fra2gt_3,139). happensAt(walking(fra2gt_3),139). orientation(fra2gt_5,150,139). coords(fra2gt_5,27,125,139).  visible(fra2gt_5,139). happensAt(walking(fra2gt_5),139).
orientation(fra2gt_3,153,140). coords(fra2gt_3,29,206,140).  visible(fra2gt_3,140). happensAt(walking(fra2gt_3),140). orientation(fra2gt_5,150,140). coords(fra2gt_5,27,125,140).  visible(fra2gt_5,140). happensAt(walking(fra2gt_5),140).
orientation(fra2gt_3,155,141). coords(fra2gt_3,29,206,141).  visible(fra2gt_3,141). happensAt(walking(fra2gt_3),141). orientation(fra2gt_5,150,141). coords(fra2gt_5,28,124,141).  visible(fra2gt_5,141). happensAt(walking(fra2gt_5),141).
orientation(fra2gt_3,150,142). coords(fra2gt_3,29,206,142).  visible(fra2gt_3,142). happensAt(walking(fra2gt_3),142). orientation(fra2gt_5,150,142). coords(fra2gt_5,28,124,142).  visible(fra2gt_5,142). happensAt(walking(fra2gt_5),142).
orientation(fra2gt_3,150,143). coords(fra2gt_3,29,207,143).  visible(fra2gt_3,143). happensAt(walking(fra2gt_3),143). orientation(fra2gt_5,150,143). coords(fra2gt_5,28,124,143).  visible(fra2gt_5,143). happensAt(walking(fra2gt_5),143).
orientation(fra2gt_3,150,144). coords(fra2gt_3,30,206,144).  visible(fra2gt_3,144). happensAt(walking(fra2gt_3),144). orientation(fra2gt_5,150,144). coords(fra2gt_5,28,123,144).  visible(fra2gt_5,144). happensAt(walking(fra2gt_5),144).
orientation(fra2gt_3,150,145). coords(fra2gt_3,30,206,145).  visible(fra2gt_3,145). happensAt(walking(fra2gt_3),145). orientation(fra2gt_5,150,145). coords(fra2gt_5,28,123,145).  visible(fra2gt_5,145). happensAt(walking(fra2gt_5),145).
orientation(fra2gt_3,150,146). coords(fra2gt_3,31,206,146).  visible(fra2gt_3,146). happensAt(walking(fra2gt_3),146). orientation(fra2gt_5,150,146). coords(fra2gt_5,28,123,146).  visible(fra2gt_5,146). happensAt(walking(fra2gt_5),146).
orientation(fra2gt_3,165,147). coords(fra2gt_3,32,206,147).  visible(fra2gt_3,147). happensAt(walking(fra2gt_3),147). orientation(fra2gt_5,150,147). coords(fra2gt_5,28,121,147).  visible(fra2gt_5,147). happensAt(walking(fra2gt_5),147).
orientation(fra2gt_3,165,148). coords(fra2gt_3,34,205,148).  visible(fra2gt_3,148). happensAt(walking(fra2gt_3),148). orientation(fra2gt_5,150,148). coords(fra2gt_5,28,121,148).  visible(fra2gt_5,148). happensAt(walking(fra2gt_5),148).
orientation(fra2gt_3,165,149). coords(fra2gt_3,34,205,149).  visible(fra2gt_3,149). happensAt(walking(fra2gt_3),149). orientation(fra2gt_5,150,149). coords(fra2gt_5,29,120,149).  visible(fra2gt_5,149). happensAt(walking(fra2gt_5),149).
orientation(fra2gt_3,165,150). coords(fra2gt_3,34,204,150).  visible(fra2gt_3,150). happensAt(walking(fra2gt_3),150). orientation(fra2gt_5,150,150). coords(fra2gt_5,29,120,150).  visible(fra2gt_5,150). happensAt(walking(fra2gt_5),150).
orientation(fra2gt_3,165,151). coords(fra2gt_3,34,204,151).  visible(fra2gt_3,151). happensAt(walking(fra2gt_3),151). orientation(fra2gt_5,150,151). coords(fra2gt_5,29,120,151).  visible(fra2gt_5,151). happensAt(walking(fra2gt_5),151).
orientation(fra2gt_3,165,152). coords(fra2gt_3,34,203,152).  visible(fra2gt_3,152). happensAt(walking(fra2gt_3),152). orientation(fra2gt_5,150,152). coords(fra2gt_5,29,120,152).  visible(fra2gt_5,152). happensAt(walking(fra2gt_5),152).
orientation(fra2gt_3,165,153). coords(fra2gt_3,34,203,153).  visible(fra2gt_3,153). happensAt(walking(fra2gt_3),153). orientation(fra2gt_5,150,153). coords(fra2gt_5,29,120,153).  visible(fra2gt_5,153). happensAt(walking(fra2gt_5),153).
orientation(fra2gt_3,165,154). coords(fra2gt_3,34,202,154).  visible(fra2gt_3,154). happensAt(walking(fra2gt_3),154). orientation(fra2gt_5,150,154). coords(fra2gt_5,29,120,154).  visible(fra2gt_5,154). happensAt(walking(fra2gt_5),154).
orientation(fra2gt_3,165,155). coords(fra2gt_3,34,202,155).  visible(fra2gt_3,155). happensAt(walking(fra2gt_3),155). orientation(fra2gt_5,150,155). coords(fra2gt_5,29,120,155).  visible(fra2gt_5,155). happensAt(walking(fra2gt_5),155).
orientation(fra2gt_3,165,156). coords(fra2gt_3,34,201,156).  visible(fra2gt_3,156). happensAt(walking(fra2gt_3),156). orientation(fra2gt_5,150,156). coords(fra2gt_5,29,120,156).  visible(fra2gt_5,156). happensAt(walking(fra2gt_5),156).
orientation(fra2gt_3,165,157). coords(fra2gt_3,34,201,157).  visible(fra2gt_3,157). happensAt(walking(fra2gt_3),157). orientation(fra2gt_5,150,157). coords(fra2gt_5,29,118,157).  visible(fra2gt_5,157). happensAt(walking(fra2gt_5),157).
orientation(fra2gt_3,165,158). coords(fra2gt_3,34,201,158).  visible(fra2gt_3,158). happensAt(walking(fra2gt_3),158). orientation(fra2gt_5,150,158). coords(fra2gt_5,29,118,158).  visible(fra2gt_5,158). happensAt(walking(fra2gt_5),158).
orientation(fra2gt_3,165,159). coords(fra2gt_3,34,200,159).  visible(fra2gt_3,159). happensAt(walking(fra2gt_3),159). orientation(fra2gt_5,150,159). coords(fra2gt_5,29,118,159).  visible(fra2gt_5,159). happensAt(walking(fra2gt_5),159).
orientation(fra2gt_3,165,160). coords(fra2gt_3,35,199,160).  visible(fra2gt_3,160). happensAt(walking(fra2gt_3),160). orientation(fra2gt_5,150,160). coords(fra2gt_5,29,117,160).  visible(fra2gt_5,160). happensAt(walking(fra2gt_5),160).
orientation(fra2gt_3,165,161). coords(fra2gt_3,35,199,161).  visible(fra2gt_3,161). happensAt(walking(fra2gt_3),161). orientation(fra2gt_5,150,161). coords(fra2gt_5,29,117,161).  visible(fra2gt_5,161). happensAt(walking(fra2gt_5),161).
orientation(fra2gt_3,165,162). coords(fra2gt_3,35,198,162).  visible(fra2gt_3,162). happensAt(walking(fra2gt_3),162). orientation(fra2gt_5,150,162). coords(fra2gt_5,29,117,162).  visible(fra2gt_5,162). happensAt(walking(fra2gt_5),162).
orientation(fra2gt_3,165,163). coords(fra2gt_3,36,198,163).  visible(fra2gt_3,163). happensAt(walking(fra2gt_3),163). orientation(fra2gt_5,150,163). coords(fra2gt_5,29,117,163).  visible(fra2gt_5,163). happensAt(walking(fra2gt_5),163).
#show holdsAt/2.