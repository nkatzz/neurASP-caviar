
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

	orientation(rwgt_0,147,123). coords(rwgt_0,202,148,123).  visible(rwgt_0,123). happensAt(walking(rwgt_0),123). orientation(rwgt_1,144,123). coords(rwgt_1,157,114,123).  visible(rwgt_1,123). happensAt(walking(rwgt_1),123).
orientation(rwgt_0,143,124). coords(rwgt_0,201,145,124).  visible(rwgt_0,124). happensAt(walking(rwgt_0),124). orientation(rwgt_1,145,124). coords(rwgt_1,157,112,124).  visible(rwgt_1,124). happensAt(walking(rwgt_1),124).
orientation(rwgt_0,143,125). coords(rwgt_0,200,141,125).  visible(rwgt_0,125). happensAt(walking(rwgt_0),125). orientation(rwgt_1,142,125). coords(rwgt_1,156,111,125).  visible(rwgt_1,125). happensAt(walking(rwgt_1),125).
orientation(rwgt_0,141,126). coords(rwgt_0,200,139,126).  visible(rwgt_0,126). happensAt(walking(rwgt_0),126). orientation(rwgt_1,138,126). coords(rwgt_1,155,110,126).  visible(rwgt_1,126). happensAt(walking(rwgt_1),126).
orientation(rwgt_0,141,127). coords(rwgt_0,198,136,127).  visible(rwgt_0,127). happensAt(walking(rwgt_0),127). orientation(rwgt_1,138,127). coords(rwgt_1,154,108,127).  visible(rwgt_1,127). happensAt(walking(rwgt_1),127).
orientation(rwgt_0,141,128). coords(rwgt_0,198,133,128).  visible(rwgt_0,128). happensAt(walking(rwgt_0),128). orientation(rwgt_1,138,128). coords(rwgt_1,152,106,128).  visible(rwgt_1,128). happensAt(walking(rwgt_1),128).
orientation(rwgt_0,141,129). coords(rwgt_0,197,131,129).  visible(rwgt_0,129). happensAt(walking(rwgt_0),129). orientation(rwgt_1,138,129). coords(rwgt_1,152,104,129).  visible(rwgt_1,129). happensAt(walking(rwgt_1),129).
orientation(rwgt_0,145,130). coords(rwgt_0,197,129,130).  visible(rwgt_0,130). happensAt(walking(rwgt_0),130). orientation(rwgt_1,138,130). coords(rwgt_1,152,101,130).  visible(rwgt_1,130). happensAt(walking(rwgt_1),130).
orientation(rwgt_0,145,131). coords(rwgt_0,196,129,131).  visible(rwgt_0,131). happensAt(walking(rwgt_0),131). orientation(rwgt_1,138,131). coords(rwgt_1,151,98,131).  visible(rwgt_1,131). happensAt(walking(rwgt_1),131).
orientation(rwgt_0,145,132). coords(rwgt_0,196,128,132).  visible(rwgt_0,132). happensAt(walking(rwgt_0),132). orientation(rwgt_1,138,132). coords(rwgt_1,151,97,132).  visible(rwgt_1,132). happensAt(walking(rwgt_1),132).
orientation(rwgt_0,145,133). coords(rwgt_0,196,128,133).  visible(rwgt_0,133). happensAt(walking(rwgt_0),133). orientation(rwgt_1,138,133). coords(rwgt_1,151,96,133).  visible(rwgt_1,133). happensAt(walking(rwgt_1),133).
orientation(rwgt_0,145,134). coords(rwgt_0,196,128,134).  visible(rwgt_0,134). happensAt(walking(rwgt_0),134). orientation(rwgt_1,138,134). coords(rwgt_1,151,96,134).  visible(rwgt_1,134). happensAt(walking(rwgt_1),134).
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
#show holdsAt/2.