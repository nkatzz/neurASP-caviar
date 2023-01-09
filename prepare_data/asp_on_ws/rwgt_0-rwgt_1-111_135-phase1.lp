
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

	orientation(rwgt_0,147,111). coords(rwgt_0,219,170,111).  visible(rwgt_0,111). happensAt(walking(rwgt_0),111). orientation(rwgt_1,149,111). coords(rwgt_1,168,132,111).  visible(rwgt_1,111). happensAt(walking(rwgt_1),111).
orientation(rwgt_0,147,112). coords(rwgt_0,218,168,112).  visible(rwgt_0,112). happensAt(walking(rwgt_0),112). orientation(rwgt_1,149,112). coords(rwgt_1,168,130,112).  visible(rwgt_1,112). happensAt(walking(rwgt_1),112).
orientation(rwgt_0,147,113). coords(rwgt_0,216,167,113).  visible(rwgt_0,113). happensAt(walking(rwgt_0),113). orientation(rwgt_1,149,113). coords(rwgt_1,167,128,113).  visible(rwgt_1,113). happensAt(walking(rwgt_1),113).
orientation(rwgt_0,147,114). coords(rwgt_0,214,165,114).  visible(rwgt_0,114). happensAt(walking(rwgt_0),114). orientation(rwgt_1,149,114). coords(rwgt_1,165,126,114).  visible(rwgt_1,114). happensAt(walking(rwgt_1),114).
orientation(rwgt_0,147,115). coords(rwgt_0,212,162,115).  visible(rwgt_0,115). happensAt(walking(rwgt_0),115). orientation(rwgt_1,149,115). coords(rwgt_1,163,123,115).  visible(rwgt_1,115). happensAt(walking(rwgt_1),115).
orientation(rwgt_0,147,116). coords(rwgt_0,210,159,116).  visible(rwgt_0,116). happensAt(walking(rwgt_0),116). orientation(rwgt_1,149,116). coords(rwgt_1,162,121,116).  visible(rwgt_1,116). happensAt(walking(rwgt_1),116).
orientation(rwgt_0,147,117). coords(rwgt_0,207,157,117).  visible(rwgt_0,117). happensAt(walking(rwgt_0),117). orientation(rwgt_1,149,117). coords(rwgt_1,160,119,117).  visible(rwgt_1,117). happensAt(walking(rwgt_1),117).
orientation(rwgt_0,147,118). coords(rwgt_0,205,156,118).  visible(rwgt_0,118). happensAt(walking(rwgt_0),118). orientation(rwgt_1,149,118). coords(rwgt_1,160,117,118).  visible(rwgt_1,118). happensAt(walking(rwgt_1),118).
orientation(rwgt_0,147,119). coords(rwgt_0,205,155,119).  visible(rwgt_0,119). happensAt(walking(rwgt_0),119). orientation(rwgt_1,149,119). coords(rwgt_1,159,117,119).  visible(rwgt_1,119). happensAt(walking(rwgt_1),119).
orientation(rwgt_0,147,120). coords(rwgt_0,204,154,120).  visible(rwgt_0,120). happensAt(walking(rwgt_0),120). orientation(rwgt_1,149,120). coords(rwgt_1,158,116,120).  visible(rwgt_1,120). happensAt(walking(rwgt_1),120).
orientation(rwgt_0,147,121). coords(rwgt_0,202,154,121).  visible(rwgt_0,121). happensAt(walking(rwgt_0),121). orientation(rwgt_1,149,121). coords(rwgt_1,157,115,121).  visible(rwgt_1,121). happensAt(walking(rwgt_1),121).
orientation(rwgt_0,147,122). coords(rwgt_0,202,150,122).  visible(rwgt_0,122). happensAt(walking(rwgt_0),122). orientation(rwgt_1,147,122). coords(rwgt_1,157,114,122).  visible(rwgt_1,122). happensAt(walking(rwgt_1),122).
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
#show holdsAt/2.