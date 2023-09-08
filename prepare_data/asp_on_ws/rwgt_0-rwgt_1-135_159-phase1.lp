
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

	orientation(rwgt_0,147,135). coords(rwgt_0,194,127,135).  visible(rwgt_0,135). happensAt(walking(rwgt_0),135). orientation(rwgt_1,138,135). coords(rwgt_1,150,95,135).  visible(rwgt_1,135). happensAt(walking(rwgt_1),135).
orientation(rwgt_0,147,136). coords(rwgt_0,195,125,136).  visible(rwgt_0,136). happensAt(walking(rwgt_0),136). orientation(rwgt_1,138,136). coords(rwgt_1,150,95,136).  visible(rwgt_1,136). happensAt(walking(rwgt_1),136).
orientation(rwgt_0,147,137). coords(rwgt_0,194,126,137).  visible(rwgt_0,137). happensAt(active(rwgt_0),137). orientation(rwgt_1,138,137). coords(rwgt_1,150,94,137).  visible(rwgt_1,137). happensAt(walking(rwgt_1),137).
orientation(rwgt_0,147,138). coords(rwgt_0,192,125,138).  visible(rwgt_0,138). happensAt(active(rwgt_0),138). orientation(rwgt_1,138,138). coords(rwgt_1,149,92,138).  visible(rwgt_1,138). happensAt(walking(rwgt_1),138).
orientation(rwgt_0,147,139). coords(rwgt_0,190,125,139).  visible(rwgt_0,139). happensAt(active(rwgt_0),139). orientation(rwgt_1,138,139). coords(rwgt_1,149,91,139).  visible(rwgt_1,139). happensAt(walking(rwgt_1),139).
orientation(rwgt_0,144,140). coords(rwgt_0,186,125,140).  visible(rwgt_0,140). happensAt(active(rwgt_0),140). orientation(rwgt_1,138,140). coords(rwgt_1,148,89,140).  visible(rwgt_1,140). happensAt(walking(rwgt_1),140).
orientation(rwgt_0,143,141). coords(rwgt_0,182,125,141).  visible(rwgt_0,141). happensAt(active(rwgt_0),141). orientation(rwgt_1,138,141). coords(rwgt_1,146,87,141).  visible(rwgt_1,141). happensAt(walking(rwgt_1),141).
orientation(rwgt_0,141,142). coords(rwgt_0,181,125,142).  visible(rwgt_0,142). happensAt(active(rwgt_0),142). orientation(rwgt_1,138,142). coords(rwgt_1,146,85,142).  visible(rwgt_1,142). happensAt(walking(rwgt_1),142).
orientation(rwgt_0,141,143). coords(rwgt_0,180,124,143).  visible(rwgt_0,143). happensAt(active(rwgt_0),143). orientation(rwgt_1,138,143). coords(rwgt_1,145,84,143).  visible(rwgt_1,143). happensAt(walking(rwgt_1),143).
orientation(rwgt_0,142,144). coords(rwgt_0,179,124,144).  visible(rwgt_0,144). happensAt(active(rwgt_0),144). orientation(rwgt_1,138,144). coords(rwgt_1,145,83,144).  visible(rwgt_1,144). happensAt(walking(rwgt_1),144).
orientation(rwgt_0,142,145). coords(rwgt_0,179,123,145).  visible(rwgt_0,145). happensAt(active(rwgt_0),145). orientation(rwgt_1,138,145). coords(rwgt_1,144,82,145).  visible(rwgt_1,145). happensAt(walking(rwgt_1),145).
orientation(rwgt_0,142,146). coords(rwgt_0,178,122,146).  visible(rwgt_0,146). happensAt(active(rwgt_0),146). orientation(rwgt_1,138,146). coords(rwgt_1,144,81,146).  visible(rwgt_1,146). happensAt(walking(rwgt_1),146).
orientation(rwgt_0,142,147). coords(rwgt_0,178,121,147).  visible(rwgt_0,147). happensAt(active(rwgt_0),147). orientation(rwgt_1,138,147). coords(rwgt_1,144,81,147).  visible(rwgt_1,147). happensAt(walking(rwgt_1),147).
orientation(rwgt_0,144,148). coords(rwgt_0,178,120,148).  visible(rwgt_0,148). happensAt(active(rwgt_0),148). orientation(rwgt_1,140,148). coords(rwgt_1,144,80,148).  visible(rwgt_1,148). happensAt(walking(rwgt_1),148).
orientation(rwgt_0,143,149). coords(rwgt_0,178,119,149).  visible(rwgt_0,149). happensAt(active(rwgt_0),149). orientation(rwgt_1,140,149). coords(rwgt_1,144,80,149).  visible(rwgt_1,149). happensAt(walking(rwgt_1),149).
orientation(rwgt_0,147,150). coords(rwgt_0,178,119,150).  visible(rwgt_0,150). happensAt(active(rwgt_0),150). orientation(rwgt_1,135,150). coords(rwgt_1,144,79,150).  visible(rwgt_1,150). happensAt(walking(rwgt_1),150).
orientation(rwgt_0,147,151). coords(rwgt_0,178,119,151).  visible(rwgt_0,151). happensAt(active(rwgt_0),151). orientation(rwgt_1,135,151). coords(rwgt_1,143,78,151).  visible(rwgt_1,151). happensAt(walking(rwgt_1),151).
orientation(rwgt_0,149,152). coords(rwgt_0,178,119,152).  visible(rwgt_0,152). happensAt(active(rwgt_0),152). orientation(rwgt_1,132,152). coords(rwgt_1,142,76,152).  visible(rwgt_1,152). happensAt(walking(rwgt_1),152).
orientation(rwgt_0,149,153). coords(rwgt_0,178,119,153).  visible(rwgt_0,153). happensAt(active(rwgt_0),153). orientation(rwgt_1,132,153). coords(rwgt_1,142,74,153).  visible(rwgt_1,153). happensAt(walking(rwgt_1),153).
orientation(rwgt_0,149,154). coords(rwgt_0,177,118,154).  visible(rwgt_0,154). happensAt(active(rwgt_0),154). orientation(rwgt_1,132,154). coords(rwgt_1,142,73,154).  visible(rwgt_1,154). happensAt(walking(rwgt_1),154).
orientation(rwgt_0,149,155). coords(rwgt_0,176,118,155).  visible(rwgt_0,155). happensAt(active(rwgt_0),155). orientation(rwgt_1,135,155). coords(rwgt_1,142,71,155).  visible(rwgt_1,155). happensAt(walking(rwgt_1),155).
orientation(rwgt_0,149,156). coords(rwgt_0,175,118,156).  visible(rwgt_0,156). happensAt(active(rwgt_0),156). orientation(rwgt_1,135,156). coords(rwgt_1,142,68,156).  visible(rwgt_1,156). happensAt(walking(rwgt_1),156).
orientation(rwgt_0,149,157). coords(rwgt_0,174,118,157).  visible(rwgt_0,157). happensAt(active(rwgt_0),157). orientation(rwgt_1,135,157). coords(rwgt_1,142,66,157).  visible(rwgt_1,157). happensAt(walking(rwgt_1),157).
orientation(rwgt_0,149,158). coords(rwgt_0,173,117,158).  visible(rwgt_0,158). happensAt(active(rwgt_0),158). orientation(rwgt_1,135,158). coords(rwgt_1,142,66,158).  visible(rwgt_1,158). happensAt(walking(rwgt_1),158).
orientation(rwgt_0,149,159). coords(rwgt_0,171,117,159).  visible(rwgt_0,159). happensAt(active(rwgt_0),159). orientation(rwgt_1,135,159). coords(rwgt_1,141,66,159).  visible(rwgt_1,159). happensAt(walking(rwgt_1),159).
#show holdsAt/2.