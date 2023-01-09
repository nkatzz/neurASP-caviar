
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

	orientation(ricgt_3,152,123). coords(ricgt_3,190,158,123).  visible(ricgt_3,123). happensAt(walking(ricgt_3),123). orientation(ricgt_4,172,123). coords(ricgt_4,26,134,123).  visible(ricgt_4,123). happensAt(walking(ricgt_4),123).
orientation(ricgt_3,153,124). coords(ricgt_3,189,156,124).  visible(ricgt_3,124). happensAt(walking(ricgt_3),124). orientation(ricgt_4,172,124). coords(ricgt_4,26,134,124).  visible(ricgt_4,124). happensAt(walking(ricgt_4),124).
orientation(ricgt_3,153,125). coords(ricgt_3,189,156,125).  visible(ricgt_3,125). happensAt(walking(ricgt_3),125). orientation(ricgt_4,172,125). coords(ricgt_4,26,134,125).  visible(ricgt_4,125). happensAt(walking(ricgt_4),125).
orientation(ricgt_3,153,126). coords(ricgt_3,189,156,126).  visible(ricgt_3,126). happensAt(walking(ricgt_3),126). orientation(ricgt_4,172,126). coords(ricgt_4,26,134,126).  visible(ricgt_4,126). happensAt(walking(ricgt_4),126).
orientation(ricgt_3,153,127). coords(ricgt_3,189,156,127).  visible(ricgt_3,127). happensAt(walking(ricgt_3),127). orientation(ricgt_4,172,127). coords(ricgt_4,26,134,127).  visible(ricgt_4,127). happensAt(walking(ricgt_4),127).
orientation(ricgt_3,154,128). coords(ricgt_3,188,156,128).  visible(ricgt_3,128). happensAt(walking(ricgt_3),128). orientation(ricgt_4,172,128). coords(ricgt_4,26,134,128).  visible(ricgt_4,128). happensAt(walking(ricgt_4),128).
orientation(ricgt_3,155,129). coords(ricgt_3,188,156,129).  visible(ricgt_3,129). happensAt(inactive(ricgt_3),129). orientation(ricgt_4,172,129). coords(ricgt_4,26,134,129).  visible(ricgt_4,129). happensAt(walking(ricgt_4),129).
orientation(ricgt_3,156,130). coords(ricgt_3,188,156,130).  visible(ricgt_3,130). happensAt(inactive(ricgt_3),130). orientation(ricgt_4,172,130). coords(ricgt_4,26,134,130).  visible(ricgt_4,130). happensAt(walking(ricgt_4),130).
orientation(ricgt_3,156,131). coords(ricgt_3,188,156,131).  visible(ricgt_3,131). happensAt(inactive(ricgt_3),131). orientation(ricgt_4,172,131). coords(ricgt_4,26,134,131).  visible(ricgt_4,131). happensAt(walking(ricgt_4),131).
orientation(ricgt_3,156,132). coords(ricgt_3,188,156,132).  visible(ricgt_3,132). happensAt(inactive(ricgt_3),132). orientation(ricgt_4,172,132). coords(ricgt_4,26,134,132).  visible(ricgt_4,132). happensAt(walking(ricgt_4),132).
orientation(ricgt_3,156,133). coords(ricgt_3,188,156,133).  visible(ricgt_3,133). happensAt(inactive(ricgt_3),133). orientation(ricgt_4,172,133). coords(ricgt_4,26,134,133).  visible(ricgt_4,133). happensAt(walking(ricgt_4),133).
orientation(ricgt_3,155,134). coords(ricgt_3,187,156,134).  visible(ricgt_3,134). happensAt(inactive(ricgt_3),134). orientation(ricgt_4,172,134). coords(ricgt_4,26,134,134).  visible(ricgt_4,134). happensAt(walking(ricgt_4),134).
orientation(ricgt_3,155,135). coords(ricgt_3,187,156,135).  visible(ricgt_3,135). happensAt(inactive(ricgt_3),135). orientation(ricgt_4,172,135). coords(ricgt_4,26,134,135).  visible(ricgt_4,135). happensAt(walking(ricgt_4),135).
orientation(ricgt_3,155,136). coords(ricgt_3,187,156,136).  visible(ricgt_3,136). happensAt(inactive(ricgt_3),136). orientation(ricgt_4,172,136). coords(ricgt_4,26,135,136).  visible(ricgt_4,136). happensAt(walking(ricgt_4),136).
orientation(ricgt_3,155,137). coords(ricgt_3,187,156,137).  visible(ricgt_3,137). happensAt(inactive(ricgt_3),137). orientation(ricgt_4,172,137). coords(ricgt_4,26,135,137).  visible(ricgt_4,137). happensAt(walking(ricgt_4),137).
orientation(ricgt_3,155,138). coords(ricgt_3,187,156,138).  visible(ricgt_3,138). happensAt(inactive(ricgt_3),138). orientation(ricgt_4,172,138). coords(ricgt_4,26,135,138).  visible(ricgt_4,138). happensAt(walking(ricgt_4),138).
orientation(ricgt_3,155,139). coords(ricgt_3,188,156,139).  visible(ricgt_3,139). happensAt(inactive(ricgt_3),139). orientation(ricgt_4,172,139). coords(ricgt_4,26,136,139).  visible(ricgt_4,139). happensAt(walking(ricgt_4),139).
orientation(ricgt_3,155,140). coords(ricgt_3,188,156,140).  visible(ricgt_3,140). happensAt(inactive(ricgt_3),140). orientation(ricgt_4,172,140). coords(ricgt_4,26,136,140).  visible(ricgt_4,140). happensAt(walking(ricgt_4),140).
orientation(ricgt_3,155,141). coords(ricgt_3,188,156,141).  visible(ricgt_3,141). happensAt(inactive(ricgt_3),141). orientation(ricgt_4,172,141). coords(ricgt_4,26,136,141).  visible(ricgt_4,141). happensAt(walking(ricgt_4),141).
orientation(ricgt_3,154,142). coords(ricgt_3,188,156,142).  visible(ricgt_3,142). happensAt(active(ricgt_3),142). orientation(ricgt_4,172,142). coords(ricgt_4,26,136,142).  visible(ricgt_4,142). happensAt(walking(ricgt_4),142).
orientation(ricgt_3,154,143). coords(ricgt_3,189,156,143).  visible(ricgt_3,143). happensAt(active(ricgt_3),143). orientation(ricgt_4,172,143). coords(ricgt_4,26,136,143).  visible(ricgt_4,143). happensAt(walking(ricgt_4),143).
orientation(ricgt_3,153,144). coords(ricgt_3,189,156,144).  visible(ricgt_3,144). happensAt(active(ricgt_3),144). orientation(ricgt_4,172,144). coords(ricgt_4,26,137,144).  visible(ricgt_4,144). happensAt(walking(ricgt_4),144).
orientation(ricgt_3,153,145). coords(ricgt_3,189,156,145).  visible(ricgt_3,145). happensAt(active(ricgt_3),145). orientation(ricgt_4,172,145). coords(ricgt_4,25,137,145).  visible(ricgt_4,145). happensAt(walking(ricgt_4),145).
orientation(ricgt_3,152,146). coords(ricgt_3,189,157,146).  visible(ricgt_3,146). happensAt(active(ricgt_3),146). orientation(ricgt_4,172,146). coords(ricgt_4,25,137,146).  visible(ricgt_4,146). happensAt(walking(ricgt_4),146).
orientation(ricgt_3,150,147). coords(ricgt_3,190,158,147).  visible(ricgt_3,147). happensAt(active(ricgt_3),147). orientation(ricgt_4,172,147). coords(ricgt_4,25,137,147).  visible(ricgt_4,147). happensAt(walking(ricgt_4),147).
#show holdsAt/2.