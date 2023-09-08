
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

	orientation(lb1gt_0,146,24). coords(lb1gt_0,171,148,24).  visible(lb1gt_0,24). happensAt(walking(lb1gt_0),24). orientation(lb1gt_1,147,24). coords(lb1gt_1,71,86,24).  visible(lb1gt_1,24). happensAt(walking(lb1gt_1),24).
orientation(lb1gt_0,146,25). coords(lb1gt_0,171,147,25).  visible(lb1gt_0,25). happensAt(walking(lb1gt_0),25). orientation(lb1gt_1,147,25). coords(lb1gt_1,71,86,25).  visible(lb1gt_1,25). happensAt(walking(lb1gt_1),25).
orientation(lb1gt_0,146,26). coords(lb1gt_0,172,146,26).  visible(lb1gt_0,26). happensAt(walking(lb1gt_0),26). orientation(lb1gt_1,147,26). coords(lb1gt_1,71,87,26).  visible(lb1gt_1,26). happensAt(walking(lb1gt_1),26).
orientation(lb1gt_0,146,27). coords(lb1gt_0,172,144,27).  visible(lb1gt_0,27). happensAt(walking(lb1gt_0),27). orientation(lb1gt_1,147,27). coords(lb1gt_1,71,87,27).  visible(lb1gt_1,27). happensAt(walking(lb1gt_1),27).
orientation(lb1gt_0,146,28). coords(lb1gt_0,173,143,28).  visible(lb1gt_0,28). happensAt(walking(lb1gt_0),28). orientation(lb1gt_1,147,28). coords(lb1gt_1,71,87,28).  visible(lb1gt_1,28). happensAt(walking(lb1gt_1),28).
orientation(lb1gt_0,146,29). coords(lb1gt_0,173,141,29).  visible(lb1gt_0,29). happensAt(walking(lb1gt_0),29). orientation(lb1gt_1,147,29). coords(lb1gt_1,71,87,29).  visible(lb1gt_1,29). happensAt(walking(lb1gt_1),29).
orientation(lb1gt_0,146,30). coords(lb1gt_0,174,138,30).  visible(lb1gt_0,30). happensAt(walking(lb1gt_0),30). orientation(lb1gt_1,147,30). coords(lb1gt_1,71,88,30).  visible(lb1gt_1,30). happensAt(walking(lb1gt_1),30).
orientation(lb1gt_0,146,31). coords(lb1gt_0,174,137,31).  visible(lb1gt_0,31). happensAt(walking(lb1gt_0),31). orientation(lb1gt_1,147,31). coords(lb1gt_1,71,88,31).  visible(lb1gt_1,31). happensAt(walking(lb1gt_1),31).
orientation(lb1gt_0,146,32). coords(lb1gt_0,175,134,32).  visible(lb1gt_0,32). happensAt(walking(lb1gt_0),32). orientation(lb1gt_1,147,32). coords(lb1gt_1,72,88,32).  visible(lb1gt_1,32). happensAt(walking(lb1gt_1),32).
orientation(lb1gt_0,149,33). coords(lb1gt_0,175,132,33).  visible(lb1gt_0,33). happensAt(walking(lb1gt_0),33). orientation(lb1gt_1,147,33). coords(lb1gt_1,72,88,33).  visible(lb1gt_1,33). happensAt(walking(lb1gt_1),33).
orientation(lb1gt_0,157,34). coords(lb1gt_0,175,130,34).  visible(lb1gt_0,34). happensAt(walking(lb1gt_0),34). orientation(lb1gt_1,147,34). coords(lb1gt_1,72,88,34).  visible(lb1gt_1,34). happensAt(walking(lb1gt_1),34).
orientation(lb1gt_0,155,35). coords(lb1gt_0,177,129,35).  visible(lb1gt_0,35). happensAt(walking(lb1gt_0),35). orientation(lb1gt_1,147,35). coords(lb1gt_1,72,88,35).  visible(lb1gt_1,35). happensAt(walking(lb1gt_1),35).
orientation(lb1gt_0,156,36). coords(lb1gt_0,177,129,36).  visible(lb1gt_0,36). happensAt(walking(lb1gt_0),36). orientation(lb1gt_1,147,36). coords(lb1gt_1,72,88,36).  visible(lb1gt_1,36). happensAt(walking(lb1gt_1),36).
orientation(lb1gt_0,150,37). coords(lb1gt_0,178,128,37).  visible(lb1gt_0,37). happensAt(walking(lb1gt_0),37). orientation(lb1gt_1,147,37). coords(lb1gt_1,73,89,37).  visible(lb1gt_1,37). happensAt(walking(lb1gt_1),37).
orientation(lb1gt_0,150,38). coords(lb1gt_0,178,128,38).  visible(lb1gt_0,38). happensAt(walking(lb1gt_0),38). orientation(lb1gt_1,147,38). coords(lb1gt_1,73,89,38).  visible(lb1gt_1,38). happensAt(walking(lb1gt_1),38).
orientation(lb1gt_0,150,39). coords(lb1gt_0,179,127,39).  visible(lb1gt_0,39). happensAt(walking(lb1gt_0),39). orientation(lb1gt_1,147,39). coords(lb1gt_1,73,90,39).  visible(lb1gt_1,39). happensAt(walking(lb1gt_1),39).
orientation(lb1gt_0,150,40). coords(lb1gt_0,180,126,40).  visible(lb1gt_0,40). happensAt(walking(lb1gt_0),40). orientation(lb1gt_1,147,40). coords(lb1gt_1,73,90,40).  visible(lb1gt_1,40). happensAt(walking(lb1gt_1),40).
orientation(lb1gt_0,150,41). coords(lb1gt_0,180,126,41).  visible(lb1gt_0,41). happensAt(walking(lb1gt_0),41). orientation(lb1gt_1,147,41). coords(lb1gt_1,73,91,41).  visible(lb1gt_1,41). happensAt(walking(lb1gt_1),41).
orientation(lb1gt_0,150,42). coords(lb1gt_0,181,124,42).  visible(lb1gt_0,42). happensAt(walking(lb1gt_0),42). orientation(lb1gt_1,147,42). coords(lb1gt_1,73,91,42).  visible(lb1gt_1,42). happensAt(walking(lb1gt_1),42).
orientation(lb1gt_0,150,43). coords(lb1gt_0,182,123,43).  visible(lb1gt_0,43). happensAt(walking(lb1gt_0),43). orientation(lb1gt_1,147,43). coords(lb1gt_1,73,91,43).  visible(lb1gt_1,43). happensAt(walking(lb1gt_1),43).
orientation(lb1gt_0,150,44). coords(lb1gt_0,182,122,44).  visible(lb1gt_0,44). happensAt(walking(lb1gt_0),44). orientation(lb1gt_1,147,44). coords(lb1gt_1,73,91,44).  visible(lb1gt_1,44). happensAt(walking(lb1gt_1),44).
orientation(lb1gt_0,148,45). coords(lb1gt_0,182,121,45).  visible(lb1gt_0,45). happensAt(walking(lb1gt_0),45). orientation(lb1gt_1,147,45). coords(lb1gt_1,73,91,45).  visible(lb1gt_1,45). happensAt(walking(lb1gt_1),45).
orientation(lb1gt_0,148,46). coords(lb1gt_0,182,121,46).  visible(lb1gt_0,46). happensAt(walking(lb1gt_0),46). orientation(lb1gt_1,147,46). coords(lb1gt_1,73,92,46).  visible(lb1gt_1,46). happensAt(walking(lb1gt_1),46).
orientation(lb1gt_0,148,47). coords(lb1gt_0,182,120,47).  visible(lb1gt_0,47). happensAt(walking(lb1gt_0),47). orientation(lb1gt_1,147,47). coords(lb1gt_1,73,92,47).  visible(lb1gt_1,47). happensAt(walking(lb1gt_1),47).
orientation(lb1gt_0,148,48). coords(lb1gt_0,182,119,48).  visible(lb1gt_0,48). happensAt(walking(lb1gt_0),48). orientation(lb1gt_1,147,48). coords(lb1gt_1,73,92,48).  visible(lb1gt_1,48). happensAt(walking(lb1gt_1),48).
#show holdsAt/2.