
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

	orientation(rsfgt_0,160,108). coords(rsfgt_0,49,153,108).  visible(rsfgt_0,108). happensAt(walking(rsfgt_0),108). orientation(rsfgt_1,150,108). coords(rsfgt_1,166,106,108).  visible(rsfgt_1,108). happensAt(active(rsfgt_1),108).
orientation(rsfgt_0,160,109). coords(rsfgt_0,49,153,109).  visible(rsfgt_0,109). happensAt(walking(rsfgt_0),109). orientation(rsfgt_1,150,109). coords(rsfgt_1,166,106,109).  visible(rsfgt_1,109). happensAt(active(rsfgt_1),109).
orientation(rsfgt_0,160,110). coords(rsfgt_0,49,153,110).  visible(rsfgt_0,110). happensAt(walking(rsfgt_0),110). orientation(rsfgt_1,150,110). coords(rsfgt_1,166,106,110).  visible(rsfgt_1,110). happensAt(active(rsfgt_1),110).
orientation(rsfgt_0,160,111). coords(rsfgt_0,48,153,111).  visible(rsfgt_0,111). happensAt(walking(rsfgt_0),111). orientation(rsfgt_1,150,111). coords(rsfgt_1,166,106,111).  visible(rsfgt_1,111). happensAt(active(rsfgt_1),111).
orientation(rsfgt_0,160,112). coords(rsfgt_0,48,153,112).  visible(rsfgt_0,112). happensAt(walking(rsfgt_0),112). orientation(rsfgt_1,150,112). coords(rsfgt_1,166,106,112).  visible(rsfgt_1,112). happensAt(active(rsfgt_1),112).
orientation(rsfgt_0,160,113). coords(rsfgt_0,48,153,113).  visible(rsfgt_0,113). happensAt(walking(rsfgt_0),113). orientation(rsfgt_1,150,113). coords(rsfgt_1,166,106,113).  visible(rsfgt_1,113). happensAt(active(rsfgt_1),113).
orientation(rsfgt_0,160,114). coords(rsfgt_0,46,153,114).  visible(rsfgt_0,114). happensAt(walking(rsfgt_0),114). orientation(rsfgt_1,150,114). coords(rsfgt_1,166,106,114).  visible(rsfgt_1,114). happensAt(active(rsfgt_1),114).
orientation(rsfgt_0,160,115). coords(rsfgt_0,45,153,115).  visible(rsfgt_0,115). happensAt(walking(rsfgt_0),115). orientation(rsfgt_1,150,115). coords(rsfgt_1,166,106,115).  visible(rsfgt_1,115). happensAt(active(rsfgt_1),115).
orientation(rsfgt_0,160,116). coords(rsfgt_0,45,153,116).  visible(rsfgt_0,116). happensAt(walking(rsfgt_0),116). orientation(rsfgt_1,150,116). coords(rsfgt_1,166,106,116).  visible(rsfgt_1,116). happensAt(active(rsfgt_1),116).
orientation(rsfgt_0,160,117). coords(rsfgt_0,45,154,117).  visible(rsfgt_0,117). happensAt(walking(rsfgt_0),117). orientation(rsfgt_1,150,117). coords(rsfgt_1,166,106,117).  visible(rsfgt_1,117). happensAt(active(rsfgt_1),117).
orientation(rsfgt_0,160,118). coords(rsfgt_0,45,154,118).  visible(rsfgt_0,118). happensAt(walking(rsfgt_0),118). orientation(rsfgt_1,150,118). coords(rsfgt_1,166,106,118).  visible(rsfgt_1,118). happensAt(active(rsfgt_1),118).
orientation(rsfgt_0,160,119). coords(rsfgt_0,44,155,119).  visible(rsfgt_0,119). happensAt(walking(rsfgt_0),119). orientation(rsfgt_1,150,119). coords(rsfgt_1,166,106,119).  visible(rsfgt_1,119). happensAt(active(rsfgt_1),119).
orientation(rsfgt_0,160,120). coords(rsfgt_0,44,155,120).  visible(rsfgt_0,120). happensAt(walking(rsfgt_0),120). orientation(rsfgt_1,150,120). coords(rsfgt_1,166,106,120).  visible(rsfgt_1,120). happensAt(active(rsfgt_1),120).
orientation(rsfgt_0,160,121). coords(rsfgt_0,44,155,121).  visible(rsfgt_0,121). happensAt(walking(rsfgt_0),121). orientation(rsfgt_1,150,121). coords(rsfgt_1,166,106,121).  visible(rsfgt_1,121). happensAt(active(rsfgt_1),121).
orientation(rsfgt_0,160,122). coords(rsfgt_0,44,155,122).  visible(rsfgt_0,122). happensAt(walking(rsfgt_0),122). orientation(rsfgt_1,150,122). coords(rsfgt_1,166,106,122).  visible(rsfgt_1,122). happensAt(active(rsfgt_1),122).
orientation(rsfgt_0,160,123). coords(rsfgt_0,43,155,123).  visible(rsfgt_0,123). happensAt(walking(rsfgt_0),123). orientation(rsfgt_1,150,123). coords(rsfgt_1,166,106,123).  visible(rsfgt_1,123). happensAt(active(rsfgt_1),123).
orientation(rsfgt_0,160,124). coords(rsfgt_0,43,155,124).  visible(rsfgt_0,124). happensAt(walking(rsfgt_0),124). orientation(rsfgt_1,150,124). coords(rsfgt_1,166,107,124).  visible(rsfgt_1,124). happensAt(active(rsfgt_1),124).
orientation(rsfgt_0,160,125). coords(rsfgt_0,42,155,125).  visible(rsfgt_0,125). happensAt(walking(rsfgt_0),125). orientation(rsfgt_1,150,125). coords(rsfgt_1,166,107,125).  visible(rsfgt_1,125). happensAt(active(rsfgt_1),125).
orientation(rsfgt_0,160,126). coords(rsfgt_0,41,155,126).  visible(rsfgt_0,126). happensAt(walking(rsfgt_0),126). orientation(rsfgt_1,150,126). coords(rsfgt_1,166,108,126).  visible(rsfgt_1,126). happensAt(active(rsfgt_1),126).
orientation(rsfgt_0,160,127). coords(rsfgt_0,41,155,127).  visible(rsfgt_0,127). happensAt(walking(rsfgt_0),127). orientation(rsfgt_1,150,127). coords(rsfgt_1,166,108,127).  visible(rsfgt_1,127). happensAt(active(rsfgt_1),127).
orientation(rsfgt_0,160,128). coords(rsfgt_0,41,155,128).  visible(rsfgt_0,128). happensAt(walking(rsfgt_0),128). orientation(rsfgt_1,150,128). coords(rsfgt_1,166,108,128).  visible(rsfgt_1,128). happensAt(active(rsfgt_1),128).
orientation(rsfgt_0,160,129). coords(rsfgt_0,41,155,129).  visible(rsfgt_0,129). happensAt(walking(rsfgt_0),129). orientation(rsfgt_1,150,129). coords(rsfgt_1,166,108,129).  visible(rsfgt_1,129). happensAt(active(rsfgt_1),129).
orientation(rsfgt_0,160,130). coords(rsfgt_0,41,155,130).  visible(rsfgt_0,130). happensAt(walking(rsfgt_0),130). orientation(rsfgt_1,150,130). coords(rsfgt_1,166,108,130).  visible(rsfgt_1,130). happensAt(active(rsfgt_1),130).
orientation(rsfgt_0,160,131). coords(rsfgt_0,41,155,131).  visible(rsfgt_0,131). happensAt(walking(rsfgt_0),131). orientation(rsfgt_1,150,131). coords(rsfgt_1,166,108,131).  visible(rsfgt_1,131). happensAt(active(rsfgt_1),131).
orientation(rsfgt_0,160,132). coords(rsfgt_0,41,155,132).  visible(rsfgt_0,132). happensAt(walking(rsfgt_0),132). orientation(rsfgt_1,150,132). coords(rsfgt_1,166,108,132).  visible(rsfgt_1,132). happensAt(active(rsfgt_1),132).
#show holdsAt/2.