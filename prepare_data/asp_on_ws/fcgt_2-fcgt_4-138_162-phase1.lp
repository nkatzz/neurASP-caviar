
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

	orientation(fcgt_2,160,138). coords(fcgt_2,23,211,138).  happensAt(appear(fcgt_2),138). happensAt(walking(fcgt_2),138). orientation(fcgt_4,170,138). coords(fcgt_4,101,210,138).  visible(fcgt_4,138). happensAt(inactive(fcgt_4),138).
orientation(fcgt_2,160,139). coords(fcgt_2,23,211,139).  visible(fcgt_2,139). happensAt(walking(fcgt_2),139). orientation(fcgt_4,170,139). coords(fcgt_4,101,210,139).  visible(fcgt_4,139). happensAt(inactive(fcgt_4),139).
orientation(fcgt_2,160,140). coords(fcgt_2,23,211,140).  visible(fcgt_2,140). happensAt(walking(fcgt_2),140). orientation(fcgt_4,170,140). coords(fcgt_4,101,210,140).  visible(fcgt_4,140). happensAt(inactive(fcgt_4),140).
orientation(fcgt_2,160,141). coords(fcgt_2,23,211,141).  visible(fcgt_2,141). happensAt(walking(fcgt_2),141). orientation(fcgt_4,170,141). coords(fcgt_4,101,210,141).  visible(fcgt_4,141). happensAt(inactive(fcgt_4),141).
orientation(fcgt_2,160,142). coords(fcgt_2,23,211,142).  visible(fcgt_2,142). happensAt(walking(fcgt_2),142). orientation(fcgt_4,170,142). coords(fcgt_4,101,210,142).  visible(fcgt_4,142). happensAt(inactive(fcgt_4),142).
orientation(fcgt_2,160,143). coords(fcgt_2,24,211,143).  visible(fcgt_2,143). happensAt(walking(fcgt_2),143). orientation(fcgt_4,170,143). coords(fcgt_4,101,210,143).  visible(fcgt_4,143). happensAt(inactive(fcgt_4),143).
orientation(fcgt_2,160,144). coords(fcgt_2,24,211,144).  visible(fcgt_2,144). happensAt(walking(fcgt_2),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
orientation(fcgt_2,160,145). coords(fcgt_2,24,211,145).  visible(fcgt_2,145). happensAt(walking(fcgt_2),145). orientation(fcgt_4,170,145). coords(fcgt_4,101,210,145).  visible(fcgt_4,145). happensAt(inactive(fcgt_4),145).
orientation(fcgt_2,160,146). coords(fcgt_2,24,211,146).  visible(fcgt_2,146). happensAt(walking(fcgt_2),146). orientation(fcgt_4,170,146). coords(fcgt_4,101,210,146).  visible(fcgt_4,146). happensAt(inactive(fcgt_4),146).
orientation(fcgt_2,160,147). coords(fcgt_2,24,211,147).  visible(fcgt_2,147). happensAt(walking(fcgt_2),147). orientation(fcgt_4,170,147). coords(fcgt_4,101,210,147).  visible(fcgt_4,147). happensAt(inactive(fcgt_4),147).
orientation(fcgt_2,160,148). coords(fcgt_2,24,211,148).  visible(fcgt_2,148). happensAt(walking(fcgt_2),148). orientation(fcgt_4,170,148). coords(fcgt_4,101,210,148).  visible(fcgt_4,148). happensAt(inactive(fcgt_4),148).
orientation(fcgt_2,160,149). coords(fcgt_2,24,210,149).  visible(fcgt_2,149). happensAt(walking(fcgt_2),149). orientation(fcgt_4,170,149). coords(fcgt_4,101,210,149).  visible(fcgt_4,149). happensAt(inactive(fcgt_4),149).
orientation(fcgt_2,160,150). coords(fcgt_2,25,209,150).  visible(fcgt_2,150). happensAt(walking(fcgt_2),150). orientation(fcgt_4,170,150). coords(fcgt_4,101,210,150).  visible(fcgt_4,150). happensAt(walking(fcgt_4),150).
orientation(fcgt_2,160,151). coords(fcgt_2,25,209,151).  visible(fcgt_2,151). happensAt(walking(fcgt_2),151). orientation(fcgt_4,170,151). coords(fcgt_4,101,210,151).  visible(fcgt_4,151). happensAt(walking(fcgt_4),151).
orientation(fcgt_2,160,152). coords(fcgt_2,25,208,152).  visible(fcgt_2,152). happensAt(walking(fcgt_2),152). orientation(fcgt_4,170,152). coords(fcgt_4,101,210,152).  visible(fcgt_4,152). happensAt(walking(fcgt_4),152).
orientation(fcgt_2,160,153). coords(fcgt_2,26,208,153).  visible(fcgt_2,153). happensAt(walking(fcgt_2),153). orientation(fcgt_4,170,153). coords(fcgt_4,101,210,153).  visible(fcgt_4,153). happensAt(walking(fcgt_4),153).
orientation(fcgt_2,160,154). coords(fcgt_2,28,207,154).  visible(fcgt_2,154). happensAt(walking(fcgt_2),154). orientation(fcgt_4,170,154). coords(fcgt_4,101,210,154).  visible(fcgt_4,154). happensAt(walking(fcgt_4),154).
orientation(fcgt_2,160,155). coords(fcgt_2,28,208,155).  visible(fcgt_2,155). happensAt(walking(fcgt_2),155). orientation(fcgt_4,170,155). coords(fcgt_4,100,210,155).  visible(fcgt_4,155). happensAt(walking(fcgt_4),155).
orientation(fcgt_2,160,156). coords(fcgt_2,28,208,156).  visible(fcgt_2,156). happensAt(walking(fcgt_2),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
orientation(fcgt_2,160,157). coords(fcgt_2,28,208,157).  visible(fcgt_2,157). happensAt(walking(fcgt_2),157). orientation(fcgt_4,170,157). coords(fcgt_4,100,210,157).  visible(fcgt_4,157). happensAt(walking(fcgt_4),157).
orientation(fcgt_2,160,158). coords(fcgt_2,28,208,158).  visible(fcgt_2,158). happensAt(walking(fcgt_2),158). orientation(fcgt_4,170,158). coords(fcgt_4,100,210,158).  visible(fcgt_4,158). happensAt(walking(fcgt_4),158).
orientation(fcgt_2,160,159). coords(fcgt_2,28,208,159).  visible(fcgt_2,159). happensAt(walking(fcgt_2),159). orientation(fcgt_4,170,159). coords(fcgt_4,100,210,159).  visible(fcgt_4,159). happensAt(walking(fcgt_4),159).
orientation(fcgt_2,160,160). coords(fcgt_2,28,208,160).  visible(fcgt_2,160). happensAt(walking(fcgt_2),160). orientation(fcgt_4,170,160). coords(fcgt_4,100,210,160).  visible(fcgt_4,160). happensAt(walking(fcgt_4),160).
orientation(fcgt_2,160,161). coords(fcgt_2,28,207,161).  visible(fcgt_2,161). happensAt(walking(fcgt_2),161). orientation(fcgt_4,170,161). coords(fcgt_4,100,210,161).  visible(fcgt_4,161). happensAt(walking(fcgt_4),161).
orientation(fcgt_2,160,162). coords(fcgt_2,28,207,162).  visible(fcgt_2,162). happensAt(walking(fcgt_2),162). orientation(fcgt_4,165,162). coords(fcgt_4,99,210,162).  visible(fcgt_4,162). happensAt(walking(fcgt_4),162).
#show holdsAt/2.