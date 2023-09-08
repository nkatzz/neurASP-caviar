
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

	orientation(mws1gt_1,170,25). coords(mws1gt_1,65,193,25).  visible(mws1gt_1,25). happensAt(inactive(mws1gt_1),25). orientation(mws1gt_2,155,25). coords(mws1gt_2,44,126,25).  visible(mws1gt_2,25). happensAt(walking(mws1gt_2),25).
orientation(mws1gt_1,170,26). coords(mws1gt_1,65,193,26).  visible(mws1gt_1,26). happensAt(inactive(mws1gt_1),26). orientation(mws1gt_2,155,26). coords(mws1gt_2,44,126,26).  visible(mws1gt_2,26). happensAt(walking(mws1gt_2),26).
orientation(mws1gt_1,170,27). coords(mws1gt_1,65,193,27).  visible(mws1gt_1,27). happensAt(inactive(mws1gt_1),27). orientation(mws1gt_2,155,27). coords(mws1gt_2,44,127,27).  visible(mws1gt_2,27). happensAt(walking(mws1gt_2),27).
orientation(mws1gt_1,170,28). coords(mws1gt_1,65,193,28).  visible(mws1gt_1,28). happensAt(inactive(mws1gt_1),28). orientation(mws1gt_2,155,28). coords(mws1gt_2,44,128,28).  visible(mws1gt_2,28). happensAt(walking(mws1gt_2),28).
orientation(mws1gt_1,170,29). coords(mws1gt_1,65,193,29).  visible(mws1gt_1,29). happensAt(inactive(mws1gt_1),29). orientation(mws1gt_2,155,29). coords(mws1gt_2,43,129,29).  visible(mws1gt_2,29). happensAt(walking(mws1gt_2),29).
orientation(mws1gt_1,170,30). coords(mws1gt_1,65,193,30).  visible(mws1gt_1,30). happensAt(inactive(mws1gt_1),30). orientation(mws1gt_2,155,30). coords(mws1gt_2,44,130,30).  visible(mws1gt_2,30). happensAt(walking(mws1gt_2),30).
orientation(mws1gt_1,170,31). coords(mws1gt_1,65,193,31).  visible(mws1gt_1,31). happensAt(inactive(mws1gt_1),31). orientation(mws1gt_2,155,31). coords(mws1gt_2,44,130,31).  visible(mws1gt_2,31). happensAt(walking(mws1gt_2),31).
orientation(mws1gt_1,170,32). coords(mws1gt_1,65,193,32).  visible(mws1gt_1,32). happensAt(inactive(mws1gt_1),32). orientation(mws1gt_2,155,32). coords(mws1gt_2,44,131,32).  visible(mws1gt_2,32). happensAt(walking(mws1gt_2),32).
orientation(mws1gt_1,170,33). coords(mws1gt_1,65,193,33).  visible(mws1gt_1,33). happensAt(inactive(mws1gt_1),33). orientation(mws1gt_2,155,33). coords(mws1gt_2,44,132,33).  visible(mws1gt_2,33). happensAt(walking(mws1gt_2),33).
orientation(mws1gt_1,170,34). coords(mws1gt_1,65,193,34).  visible(mws1gt_1,34). happensAt(inactive(mws1gt_1),34). orientation(mws1gt_2,155,34). coords(mws1gt_2,44,133,34).  visible(mws1gt_2,34). happensAt(walking(mws1gt_2),34).
orientation(mws1gt_1,170,35). coords(mws1gt_1,65,193,35).  visible(mws1gt_1,35). happensAt(inactive(mws1gt_1),35). orientation(mws1gt_2,155,35). coords(mws1gt_2,43,134,35).  visible(mws1gt_2,35). happensAt(walking(mws1gt_2),35).
orientation(mws1gt_1,170,36). coords(mws1gt_1,65,193,36).  visible(mws1gt_1,36). happensAt(inactive(mws1gt_1),36). orientation(mws1gt_2,155,36). coords(mws1gt_2,44,135,36).  visible(mws1gt_2,36). happensAt(walking(mws1gt_2),36).
orientation(mws1gt_1,170,37). coords(mws1gt_1,65,193,37).  visible(mws1gt_1,37). happensAt(inactive(mws1gt_1),37). orientation(mws1gt_2,155,37). coords(mws1gt_2,44,136,37).  visible(mws1gt_2,37). happensAt(walking(mws1gt_2),37).
orientation(mws1gt_1,170,38). coords(mws1gt_1,65,193,38).  visible(mws1gt_1,38). happensAt(inactive(mws1gt_1),38). orientation(mws1gt_2,155,38). coords(mws1gt_2,43,137,38).  visible(mws1gt_2,38). happensAt(walking(mws1gt_2),38).
orientation(mws1gt_1,170,39). coords(mws1gt_1,65,193,39).  visible(mws1gt_1,39). happensAt(inactive(mws1gt_1),39). orientation(mws1gt_2,155,39). coords(mws1gt_2,43,138,39).  visible(mws1gt_2,39). happensAt(walking(mws1gt_2),39).
orientation(mws1gt_1,170,40). coords(mws1gt_1,65,193,40).  visible(mws1gt_1,40). happensAt(inactive(mws1gt_1),40). orientation(mws1gt_2,155,40). coords(mws1gt_2,43,139,40).  visible(mws1gt_2,40). happensAt(walking(mws1gt_2),40).
orientation(mws1gt_1,170,41). coords(mws1gt_1,65,193,41).  visible(mws1gt_1,41). happensAt(inactive(mws1gt_1),41). orientation(mws1gt_2,155,41). coords(mws1gt_2,43,140,41).  visible(mws1gt_2,41). happensAt(walking(mws1gt_2),41).
orientation(mws1gt_1,170,42). coords(mws1gt_1,65,193,42).  visible(mws1gt_1,42). happensAt(inactive(mws1gt_1),42). orientation(mws1gt_2,155,42). coords(mws1gt_2,43,140,42).  visible(mws1gt_2,42). happensAt(walking(mws1gt_2),42).
orientation(mws1gt_1,170,43). coords(mws1gt_1,65,193,43).  visible(mws1gt_1,43). happensAt(inactive(mws1gt_1),43). orientation(mws1gt_2,163,43). coords(mws1gt_2,42,140,43).  visible(mws1gt_2,43). happensAt(walking(mws1gt_2),43).
orientation(mws1gt_1,170,44). coords(mws1gt_1,65,193,44).  visible(mws1gt_1,44). happensAt(inactive(mws1gt_1),44). orientation(mws1gt_2,165,44). coords(mws1gt_2,41,140,44).  visible(mws1gt_2,44). happensAt(walking(mws1gt_2),44).
orientation(mws1gt_1,170,45). coords(mws1gt_1,65,193,45).  visible(mws1gt_1,45). happensAt(inactive(mws1gt_1),45). orientation(mws1gt_2,165,45). coords(mws1gt_2,41,142,45).  visible(mws1gt_2,45). happensAt(walking(mws1gt_2),45).
orientation(mws1gt_1,170,46). coords(mws1gt_1,65,193,46).  visible(mws1gt_1,46). happensAt(inactive(mws1gt_1),46). orientation(mws1gt_2,165,46). coords(mws1gt_2,41,143,46).  visible(mws1gt_2,46). happensAt(walking(mws1gt_2),46).
orientation(mws1gt_1,170,47). coords(mws1gt_1,65,193,47).  visible(mws1gt_1,47). happensAt(inactive(mws1gt_1),47). orientation(mws1gt_2,165,47). coords(mws1gt_2,41,146,47).  visible(mws1gt_2,47). happensAt(walking(mws1gt_2),47).
orientation(mws1gt_1,170,48). coords(mws1gt_1,65,193,48).  visible(mws1gt_1,48). happensAt(inactive(mws1gt_1),48). orientation(mws1gt_2,165,48). coords(mws1gt_2,41,147,48).  visible(mws1gt_2,48). happensAt(walking(mws1gt_2),48).
orientation(mws1gt_1,170,49). coords(mws1gt_1,65,193,49).  visible(mws1gt_1,49). happensAt(inactive(mws1gt_1),49). orientation(mws1gt_2,165,49). coords(mws1gt_2,41,147,49).  visible(mws1gt_2,49). happensAt(walking(mws1gt_2),49).
#show holdsAt/2.