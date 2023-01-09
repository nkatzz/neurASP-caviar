
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

	orientation(fcgt_2,160,138). coords(fcgt_2,23,211,138).  happensAt(appear(fcgt_2),138). happensAt(walking(fcgt_2),138). orientation(fcgt_3,159,138). coords(fcgt_3,103,163,138).  visible(fcgt_3,138). happensAt(inactive(fcgt_3),138).
orientation(fcgt_2,160,139). coords(fcgt_2,23,211,139).  visible(fcgt_2,139). happensAt(walking(fcgt_2),139). orientation(fcgt_3,159,139). coords(fcgt_3,103,163,139).  visible(fcgt_3,139). happensAt(inactive(fcgt_3),139).
orientation(fcgt_2,160,140). coords(fcgt_2,23,211,140).  visible(fcgt_2,140). happensAt(walking(fcgt_2),140). orientation(fcgt_3,159,140). coords(fcgt_3,103,163,140).  visible(fcgt_3,140). happensAt(inactive(fcgt_3),140).
orientation(fcgt_2,160,141). coords(fcgt_2,23,211,141).  visible(fcgt_2,141). happensAt(walking(fcgt_2),141). orientation(fcgt_3,159,141). coords(fcgt_3,103,163,141).  visible(fcgt_3,141). happensAt(inactive(fcgt_3),141).
orientation(fcgt_2,160,142). coords(fcgt_2,23,211,142).  visible(fcgt_2,142). happensAt(walking(fcgt_2),142). orientation(fcgt_3,159,142). coords(fcgt_3,103,163,142).  visible(fcgt_3,142). happensAt(inactive(fcgt_3),142).
orientation(fcgt_2,160,143). coords(fcgt_2,24,211,143).  visible(fcgt_2,143). happensAt(walking(fcgt_2),143). orientation(fcgt_3,159,143). coords(fcgt_3,103,163,143).  visible(fcgt_3,143). happensAt(inactive(fcgt_3),143).
orientation(fcgt_2,160,144). coords(fcgt_2,24,211,144).  visible(fcgt_2,144). happensAt(walking(fcgt_2),144). orientation(fcgt_3,159,144). coords(fcgt_3,103,163,144).  visible(fcgt_3,144). happensAt(inactive(fcgt_3),144).
orientation(fcgt_2,160,145). coords(fcgt_2,24,211,145).  visible(fcgt_2,145). happensAt(walking(fcgt_2),145). orientation(fcgt_3,159,145). coords(fcgt_3,103,163,145).  visible(fcgt_3,145). happensAt(inactive(fcgt_3),145).
orientation(fcgt_2,160,146). coords(fcgt_2,24,211,146).  visible(fcgt_2,146). happensAt(walking(fcgt_2),146). orientation(fcgt_3,159,146). coords(fcgt_3,103,163,146).  visible(fcgt_3,146). happensAt(inactive(fcgt_3),146).
orientation(fcgt_2,160,147). coords(fcgt_2,24,211,147).  visible(fcgt_2,147). happensAt(walking(fcgt_2),147). orientation(fcgt_3,159,147). coords(fcgt_3,103,163,147).  visible(fcgt_3,147). happensAt(inactive(fcgt_3),147).
orientation(fcgt_2,160,148). coords(fcgt_2,24,211,148).  visible(fcgt_2,148). happensAt(walking(fcgt_2),148). orientation(fcgt_3,159,148). coords(fcgt_3,103,163,148).  visible(fcgt_3,148). happensAt(inactive(fcgt_3),148).
orientation(fcgt_2,160,149). coords(fcgt_2,24,210,149).  visible(fcgt_2,149). happensAt(walking(fcgt_2),149). orientation(fcgt_3,159,149). coords(fcgt_3,103,163,149).  visible(fcgt_3,149). happensAt(inactive(fcgt_3),149).
orientation(fcgt_2,160,150). coords(fcgt_2,25,209,150).  visible(fcgt_2,150). happensAt(walking(fcgt_2),150). orientation(fcgt_3,159,150). coords(fcgt_3,103,163,150).  visible(fcgt_3,150). happensAt(inactive(fcgt_3),150).
orientation(fcgt_2,160,151). coords(fcgt_2,25,209,151).  visible(fcgt_2,151). happensAt(walking(fcgt_2),151). orientation(fcgt_3,159,151). coords(fcgt_3,103,163,151).  visible(fcgt_3,151). happensAt(inactive(fcgt_3),151).
orientation(fcgt_2,160,152). coords(fcgt_2,25,208,152).  visible(fcgt_2,152). happensAt(walking(fcgt_2),152). orientation(fcgt_3,159,152). coords(fcgt_3,103,163,152).  visible(fcgt_3,152). happensAt(inactive(fcgt_3),152).
orientation(fcgt_2,160,153). coords(fcgt_2,26,208,153).  visible(fcgt_2,153). happensAt(walking(fcgt_2),153). orientation(fcgt_3,159,153). coords(fcgt_3,103,163,153).  visible(fcgt_3,153). happensAt(inactive(fcgt_3),153).
orientation(fcgt_2,160,154). coords(fcgt_2,28,207,154).  visible(fcgt_2,154). happensAt(walking(fcgt_2),154). orientation(fcgt_3,159,154). coords(fcgt_3,103,163,154).  visible(fcgt_3,154). happensAt(inactive(fcgt_3),154).
orientation(fcgt_2,160,155). coords(fcgt_2,28,208,155).  visible(fcgt_2,155). happensAt(walking(fcgt_2),155). orientation(fcgt_3,159,155). coords(fcgt_3,103,163,155).  visible(fcgt_3,155). happensAt(inactive(fcgt_3),155).
orientation(fcgt_2,160,156). coords(fcgt_2,28,208,156).  visible(fcgt_2,156). happensAt(walking(fcgt_2),156). orientation(fcgt_3,159,156). coords(fcgt_3,103,163,156).  visible(fcgt_3,156). happensAt(inactive(fcgt_3),156).
orientation(fcgt_2,160,157). coords(fcgt_2,28,208,157).  visible(fcgt_2,157). happensAt(walking(fcgt_2),157). orientation(fcgt_3,159,157). coords(fcgt_3,103,163,157).  visible(fcgt_3,157). happensAt(inactive(fcgt_3),157).
orientation(fcgt_2,160,158). coords(fcgt_2,28,208,158).  visible(fcgt_2,158). happensAt(walking(fcgt_2),158). orientation(fcgt_3,159,158). coords(fcgt_3,103,163,158).  visible(fcgt_3,158). happensAt(inactive(fcgt_3),158).
orientation(fcgt_2,160,159). coords(fcgt_2,28,208,159).  visible(fcgt_2,159). happensAt(walking(fcgt_2),159). orientation(fcgt_3,159,159). coords(fcgt_3,103,163,159).  visible(fcgt_3,159). happensAt(inactive(fcgt_3),159).
orientation(fcgt_2,160,160). coords(fcgt_2,28,208,160).  visible(fcgt_2,160). happensAt(walking(fcgt_2),160). orientation(fcgt_3,159,160). coords(fcgt_3,103,163,160).  visible(fcgt_3,160). happensAt(inactive(fcgt_3),160).
orientation(fcgt_2,160,161). coords(fcgt_2,28,207,161).  visible(fcgt_2,161). happensAt(walking(fcgt_2),161). orientation(fcgt_3,159,161). coords(fcgt_3,103,163,161).  visible(fcgt_3,161). happensAt(inactive(fcgt_3),161).
orientation(fcgt_2,160,162). coords(fcgt_2,28,207,162).  visible(fcgt_2,162). happensAt(walking(fcgt_2),162). orientation(fcgt_3,159,162). coords(fcgt_3,103,163,162).  visible(fcgt_3,162). happensAt(inactive(fcgt_3),162).
#show holdsAt/2.