
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

	orientation(mws1gt_3,154,138). coords(mws1gt_3,197,171,138).  visible(mws1gt_3,138). happensAt(walking(mws1gt_3),138). orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138).
orientation(mws1gt_3,154,139). coords(mws1gt_3,195,170,139).  visible(mws1gt_3,139). happensAt(walking(mws1gt_3),139). orientation(mws1gt_4,0,139). coords(mws1gt_4,27,137,139).  visible(mws1gt_4,139). happensAt(walking(mws1gt_4),139).
orientation(mws1gt_3,154,140). coords(mws1gt_3,193,168,140).  visible(mws1gt_3,140). happensAt(walking(mws1gt_3),140). orientation(mws1gt_4,0,140). coords(mws1gt_4,28,136,140).  visible(mws1gt_4,140). happensAt(walking(mws1gt_4),140).
orientation(mws1gt_3,154,141). coords(mws1gt_3,190,166,141).  visible(mws1gt_3,141). happensAt(walking(mws1gt_3),141). orientation(mws1gt_4,0,141). coords(mws1gt_4,28,136,141).  visible(mws1gt_4,141). happensAt(walking(mws1gt_4),141).
orientation(mws1gt_3,154,142). coords(mws1gt_3,188,164,142).  visible(mws1gt_3,142). happensAt(walking(mws1gt_3),142). orientation(mws1gt_4,0,142). coords(mws1gt_4,28,134,142).  visible(mws1gt_4,142). happensAt(walking(mws1gt_4),142).
orientation(mws1gt_3,154,143). coords(mws1gt_3,187,164,143).  visible(mws1gt_3,143). happensAt(walking(mws1gt_3),143). orientation(mws1gt_4,0,143). coords(mws1gt_4,28,134,143).  visible(mws1gt_4,143). happensAt(walking(mws1gt_4),143).
orientation(mws1gt_3,154,144). coords(mws1gt_3,186,163,144).  visible(mws1gt_3,144). happensAt(walking(mws1gt_3),144). orientation(mws1gt_4,0,144). coords(mws1gt_4,28,134,144).  visible(mws1gt_4,144). happensAt(walking(mws1gt_4),144).
orientation(mws1gt_3,154,145). coords(mws1gt_3,185,163,145).  visible(mws1gt_3,145). happensAt(walking(mws1gt_3),145). orientation(mws1gt_4,0,145). coords(mws1gt_4,28,134,145).  visible(mws1gt_4,145). happensAt(walking(mws1gt_4),145).
orientation(mws1gt_3,154,146). coords(mws1gt_3,185,162,146).  visible(mws1gt_3,146). happensAt(walking(mws1gt_3),146). orientation(mws1gt_4,0,146). coords(mws1gt_4,28,134,146).  visible(mws1gt_4,146). happensAt(walking(mws1gt_4),146).
orientation(mws1gt_3,154,147). coords(mws1gt_3,185,162,147).  visible(mws1gt_3,147). happensAt(walking(mws1gt_3),147). orientation(mws1gt_4,0,147). coords(mws1gt_4,28,134,147).  visible(mws1gt_4,147). happensAt(walking(mws1gt_4),147).
orientation(mws1gt_3,154,148). coords(mws1gt_3,184,161,148).  visible(mws1gt_3,148). happensAt(walking(mws1gt_3),148). orientation(mws1gt_4,0,148). coords(mws1gt_4,28,133,148).  visible(mws1gt_4,148). happensAt(walking(mws1gt_4),148).
orientation(mws1gt_3,154,149). coords(mws1gt_3,184,161,149).  visible(mws1gt_3,149). happensAt(walking(mws1gt_3),149). orientation(mws1gt_4,0,149). coords(mws1gt_4,28,133,149).  visible(mws1gt_4,149). happensAt(walking(mws1gt_4),149).
orientation(mws1gt_3,154,150). coords(mws1gt_3,183,160,150).  visible(mws1gt_3,150). happensAt(walking(mws1gt_3),150). orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150).
orientation(mws1gt_3,154,151). coords(mws1gt_3,181,160,151).  visible(mws1gt_3,151). happensAt(walking(mws1gt_3),151). orientation(mws1gt_4,0,151). coords(mws1gt_4,28,132,151).  visible(mws1gt_4,151). happensAt(walking(mws1gt_4),151).
orientation(mws1gt_3,154,152). coords(mws1gt_3,179,158,152).  visible(mws1gt_3,152). happensAt(walking(mws1gt_3),152). orientation(mws1gt_4,0,152). coords(mws1gt_4,28,132,152).  visible(mws1gt_4,152). happensAt(walking(mws1gt_4),152).
orientation(mws1gt_3,154,153). coords(mws1gt_3,178,157,153).  visible(mws1gt_3,153). happensAt(walking(mws1gt_3),153). orientation(mws1gt_4,0,153). coords(mws1gt_4,28,131,153).  visible(mws1gt_4,153). happensAt(walking(mws1gt_4),153).
orientation(mws1gt_3,154,154). coords(mws1gt_3,176,155,154).  visible(mws1gt_3,154). happensAt(walking(mws1gt_3),154). orientation(mws1gt_4,0,154). coords(mws1gt_4,28,129,154).  visible(mws1gt_4,154). happensAt(walking(mws1gt_4),154).
orientation(mws1gt_3,154,155). coords(mws1gt_3,174,154,155).  visible(mws1gt_3,155). happensAt(walking(mws1gt_3),155). orientation(mws1gt_4,0,155). coords(mws1gt_4,28,129,155).  visible(mws1gt_4,155). happensAt(walking(mws1gt_4),155).
orientation(mws1gt_3,154,156). coords(mws1gt_3,173,154,156).  visible(mws1gt_3,156). happensAt(walking(mws1gt_3),156). orientation(mws1gt_4,0,156). coords(mws1gt_4,28,127,156).  visible(mws1gt_4,156). happensAt(walking(mws1gt_4),156).
orientation(mws1gt_3,154,157). coords(mws1gt_3,171,154,157).  visible(mws1gt_3,157). happensAt(walking(mws1gt_3),157). orientation(mws1gt_4,0,157). coords(mws1gt_4,28,125,157).  visible(mws1gt_4,157). happensAt(walking(mws1gt_4),157).
orientation(mws1gt_3,154,158). coords(mws1gt_3,171,154,158).  visible(mws1gt_3,158). happensAt(walking(mws1gt_3),158). orientation(mws1gt_4,0,158). coords(mws1gt_4,28,124,158).  visible(mws1gt_4,158). happensAt(walking(mws1gt_4),158).
orientation(mws1gt_3,154,159). coords(mws1gt_3,171,154,159).  visible(mws1gt_3,159). happensAt(walking(mws1gt_3),159). orientation(mws1gt_4,0,159). coords(mws1gt_4,28,124,159).  visible(mws1gt_4,159). happensAt(walking(mws1gt_4),159).
orientation(mws1gt_3,154,160). coords(mws1gt_3,170,154,160).  visible(mws1gt_3,160). happensAt(walking(mws1gt_3),160). orientation(mws1gt_4,0,160). coords(mws1gt_4,28,124,160).  visible(mws1gt_4,160). happensAt(walking(mws1gt_4),160).
orientation(mws1gt_3,154,161). coords(mws1gt_3,170,154,161).  visible(mws1gt_3,161). happensAt(walking(mws1gt_3),161). orientation(mws1gt_4,0,161). coords(mws1gt_4,28,124,161).  visible(mws1gt_4,161). happensAt(walking(mws1gt_4),161).
orientation(mws1gt_3,154,162). coords(mws1gt_3,169,154,162).  visible(mws1gt_3,162). happensAt(walking(mws1gt_3),162). orientation(mws1gt_4,0,162). coords(mws1gt_4,28,124,162).  visible(mws1gt_4,162). happensAt(walking(mws1gt_4),162).
#show holdsAt/2.